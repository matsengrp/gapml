import time
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict
from numpy import ndarray
from scipy.stats import poisson
from tensorflow import Session

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract, AncState, SingletonWC, Singleton
from transition_matrix import TransitionMatrixWrapper, TransitionMatrix
from common import merge_target_tract_groups, not_equal_float, equal_float
from bounded_poisson import BoundedPoisson

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type
    """
    UNLIKELY = "unlikely"
    NODE_ORDER = "postorder"
    gamma_prior = (1,10)

    def __init__(self,
            topology: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            sess: Session,
            branch_lens: ndarray = None,
            target_lams: ndarray = None,
            trim_long_probs: ndarray = 0.1 * np.ones(2),
            trim_zero_prob: float = 0.5,
            trim_poissons: ndarray = np.ones(2),
            insert_zero_prob: float = 0.5,
            insert_poisson: float = 2):
            #TODO: cell_type_lams: ndarray = None):
        """
        @param topology: provides a topology only (ignore any branch lengths in this tree)
        Will randomly initialize model parameters
        """
        self.topology = topology
        self.num_nodes = 0
        if self.topology:
            node_id = 0
            for node in topology.traverse(self.NODE_ORDER):
                node.add_feature("node_id", node_id)
                if node.is_root():
                    self.root_node_id = node_id
                node_id += 1
            self.num_nodes = node_id
        self.bcode_meta = bcode_meta
        self.num_targets = bcode_meta.n_targets

        # Save tensorflow session
        self.sess = sess

        # Create all the variables
        if branch_lens is None:
            branch_lens = np.random.gamma(
                    self.gamma_prior[0],
                    self.gamma_prior[1],
                    self.num_nodes)
        if target_lams is None:
            target_lams = 0.1 * np.ones(self.num_targets)

        self.branch_lens = tf.Variable(branch_lens)
        self.target_lams = tf.Variable(target_lams)
        self.trim_long_probs = tf.Variable(trim_long_probs)
        self.trim_zero_prob = tf.Variable(trim_zero_prob)
        self.trim_poissons = tf.Variable(trim_poissons)
        self.insert_zero_prob = tf.Variable(insert_zero_prob)
        self.insert_poisson = tf.Variable(insert_poisson)

        self.grad_opt = tf.train.GradientDescentOptimizer(learning_rate=1)
        self._create_hazard()
        self._create_hazard_away()

    def _create_hazard(self):
        """
        Helpers for creating hazard tensor and its associated gradient
        """
        # Create the placeholders
        self.targets_ph = tf.placeholder(tf.int32, [None, 2])
        min_target = self.targets_ph[:,0]
        max_target = self.targets_ph[:,1]
        self.long_status_ph = tf.placeholder(tf.float64, [None, 2])
        is_left_long = self.long_status_ph[:,0]
        is_right_long = self.long_status_ph[:,1]

        # Compute the hazard
        log_lambda_part = tf.log(tf.gather(self.target_lams, min_target)) + tf.log(tf.gather(self.target_lams, max_target)) * not_equal_float(min_target, max_target)
        left_trim_prob = tf.multiply(is_left_long, self.trim_long_probs[0]) + tf.multiply(1 - is_left_long, 1 - self.trim_long_probs[0])
        right_trim_prob = tf.multiply(is_right_long, self.trim_long_probs[1]) + tf.multiply(1 - is_right_long, 1 - self.trim_long_probs[1])
        self.hazard = tf.exp(log_lambda_part + tf.log(left_trim_prob) + tf.log(right_trim_prob))

        # Compute the gradient
        self.hazard_grad = self.grad_opt.compute_gradients(
            self.hazard,
            var_list=[self.target_lams, self.trim_long_probs])

    def _create_hazard_away(self):
        """
        Helpers for creating hazard-away tensor and its associated gradient
        """
        # Create the placeholders
        self.left_trimmables_ph = tf.placeholder(tf.int32, [None, self.num_targets])
        self.right_trimmables_ph = tf.placeholder(tf.int32, [None, self.num_targets])

        # Compute the hazard away
        focal_hazards = self._create_hazard_list(True, True)
        left_hazards = self._create_hazard_list(True, False)[:, :self.num_targets - 1]
        right_hazards = self._create_hazard_list(False, True)[:, 1:]
        left_cum_hazards = tf.cumsum(left_hazards, axis=1)
        inter_target_hazards = tf.multiply(left_cum_hazards, right_hazards)
        self.hazard_away = tf.reduce_sum(focal_hazards, axis=1) + tf.reduce_sum(inter_target_hazards, axis=1)

        # Compute the gradient
        self.hazard_away_grad = self.grad_opt.compute_gradients(
            self.hazard_away,
            var_list=[self.target_lams, self.trim_long_probs])

    def _create_hazard_list(self, trim_left: bool, trim_right: bool):
        """
        Helper function for creating hazard list nodes -- useful for calculating hazard away
        """
        trim_short_left = 1 - self.trim_long_probs[0] if trim_left else 1
        trim_short_right = 1 - self.trim_long_probs[1] if trim_right else 1

        left_factor = equal_float(self.left_trimmables_ph, 1) * trim_short_left + equal_float(self.left_trimmables_ph, 2)

        right_factor = equal_float(self.right_trimmables_ph, 1) * trim_short_right + equal_float(self.right_trimmables_ph, 2)

        hazard_list = self.target_lams * left_factor * right_factor
        return hazard_list

    def initialize_transition_matrices(self, transition_mat_wrappers: Dict[int, TransitionMatrixWrapper]):
        """
        @return a list of real transition matrices
        """
        UNLIKELY = "unlikely"
        all_matrices = dict()
        for node_id, matrix_wrapper in transition_mat_wrappers.items():
            # Get inputs ready for tensorflow
            hazard_list = []
            tt_evts = []
            start_tts_list = []
            for start_tts, matrix_row in matrix_wrapper.matrix_dict.items():
                start_tts_list.append(start_tts)
                for end_tts, tt_evt in matrix_row.items():
                    tt_evts.append(tt_evt)

            # Gets hazards (by tensorflow)
            hazard_aways = self.get_hazard_aways(start_tts_list)
            hazards = self.get_hazards(tt_evts)

            # Now fill in the matrix
            idx = 0
            matrix_dict = dict()
            for i, (start_tts, matrix_row) in enumerate(matrix_wrapper.matrix_dict.items()):
                matrix_dict[start_tts] = dict()
                # Tracks the total hazard to the likely states
                haz_to_likely = 0
                for end_tts, tt_evt in matrix_row.items():
                    haz = hazards[idx]
                    matrix_dict[start_tts][end_tts] = haz
                    haz_to_likely += haz
                    idx += 1

                haz_away = hazard_aways[i]
                # Hazard to unlikely state is hazard away minus hazard to likely states
                matrix_dict[start_tts][UNLIKELY] = haz_away - haz_to_likely
                # Hazard of staying is negative of hazard away
                matrix_dict[start_tts][start_tts] = -haz_away

            # Add unlikely state
            matrix_dict[UNLIKELY] = dict()

            all_matrices[node_id] = TransitionMatrix(matrix_dict)
        return all_matrices

    def get_hazards(self, tt_evts: List[TargetTract]):
        """
        Propagates through the tensorflow graph to obtain hazard of each TargetTract in `tt_evts`
        """
        if len(tt_evts) == 0:
            return []

        target_inputs = []
        long_status_inputs = []
        for tt_evt in tt_evts:
            target_inputs.append([tt_evt.min_target, tt_evt.max_target])
            long_status_inputs.append([tt_evt.is_left_long, tt_evt.is_right_long])

        hazards = self.sess.run(self.hazard, feed_dict={
            self.targets_ph: target_inputs,
            self.long_status_ph: long_status_inputs})
        return hazards

    def get_hazard(self, tt_evt: TargetTract):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        return self.get_hazards([tt_evt])[0]

    def _get_hazard_masks(self, tts:Tuple[TargetTract]):
        """
        @param tts: the target tract repr that we would like to process

        @return two lists, one for left trims and one for right trims.
                each list is composed of values [0,1,2] matching each target.
                0 = no trim at that target (in that direction) can be performed
                1 = only short trim at that target (in that dir) is allowed
                2 = short and long trims are allowed at that target

                We assume no long left trims are allowed at target 0 and any target T
                where T - 1 is deactivated. No long right trims are allowed at the last
                target and any target T where T + 1 is deactivated.
        """
        if len(tts) == 0:
            # This is the original barcode
            left_hazard_list = [1] + [2] * (self.num_targets - 1)
            right_hazard_list = [2] * (self.num_targets - 1) + [1]
        else:
            # This is an allele with some indel somewhere

            # Process the section before the first indel
            if tts[0].min_deact_target > 1:
                left_hazard_list = [1] + [2] * (tts[0].min_deact_target - 1)
                right_hazard_list = [2] * (tts[0].min_deact_target - 1) + [1]
            elif tts[0].min_deact_target == 1:
                left_hazard_list = [1]
                right_hazard_list = [1]
            else:
                left_hazard_list = []
                right_hazard_list = []

            left_hazard_list += [0] * (tts[0].max_deact_target - tts[0].min_deact_target + 1)
            right_hazard_list += [0] * (tts[0].max_deact_target - tts[0].min_deact_target + 1)

            # Process the sections with the indels
            for i, tt in enumerate(tts[1:]):
                if tts[i].max_deact_target + 1 < tt.min_deact_target - 1:
                    left_hazard_list += [1] + [2] * (tt.min_deact_target - tts[i].max_deact_target - 2)
                    right_hazard_list += [2] * (tt.min_deact_target - tts[i].max_deact_target - 2) + [1]
                elif tts[i].max_deact_target + 1 == tt.min_deact_target - 1:
                    # Single target, short left and right
                    left_hazard_list += [1]
                    right_hazard_list += [1]
                left_hazard_list += [0] * (tt.max_deact_target - tt.min_deact_target + 1)
                right_hazard_list += [0] * (tt.max_deact_target - tt.min_deact_target + 1)

            # Process the section after all the indels
            if tts[-1].max_deact_target < self.num_targets - 2:
                left_hazard_list += [1] + [2] * (self.num_targets - tts[-1].max_deact_target - 2)
                right_hazard_list += [2] * (self.num_targets - tts[-1].max_deact_target - 2) + [1]
            elif tts[-1].max_deact_target == self.num_targets - 2:
                left_hazard_list += [1]
                right_hazard_list += [1]

        return left_hazard_list, right_hazard_list

    def get_hazard_aways(self, tts_list: List[Tuple[TargetTract]]):
        """
        Run thru the tensorflow graph to obtain the hazard of going away from these
        target tract reprs in `tts_list`
        """
        left_trimmables = []
        right_trimmables = []
        for tts in tts_list:
            left_m, right_m = self._get_hazard_masks(tts)
            left_trimmables.append(left_m)
            right_trimmables.append(right_m)

        hazard_aways = self.sess.run(
                self.hazard_away,
                feed_dict = {
                    self.left_trimmables_ph: left_trimmables,
                    self.right_trimmables_ph: right_trimmables})
        return hazard_aways

    def get_hazard_away(self, tts: Tuple[TargetTract]):
        return self.get_hazard_aways([tts])

    def get_prob_unmasked_trims(self, anc_state: AncState, tts: Tuple[TargetTract]):
        """
        @return probability of the trims for the corresponding singletons in the anc_state
        """
        matching_sgs = CLTLikelihoodModel.get_matching_singletons(anc_state, tts)
        return self._get_cond_prob_trims(matching_sgs)

    def _get_cond_prob_trims(self, singletons: List[Singleton]):
        """
        @return product of conditional probabs of the trims associated with each singleton
        """
        prob = 1
        for singleton in singletons:
            # Calculate probability of that singleton
            del_prob = self._get_cond_prob_singleton_del(singleton)
            insert_prob = self._get_cond_prob_singleton_insert(singleton)
            prob *= del_prob * insert_prob
        return prob

    @staticmethod
    def get_matching_singletons(anc_state: AncState, tts: Tuple[TargetTract]):
        """
        @return the list of singletons in `anc_state` that match any target tract in `tts`
        """
        # Get only the singletons
        sg_only_anc_state = [
            indel_set for indel_set in anc_state.indel_set_list if indel_set.__class__ == SingletonWC]

        # Now figure out which ones match
        sg_idx = 0
        tts_idx = 0
        n_sg = len(sg_only_anc_state)
        n_tts = len(tts)
        matching_sgs = []
        while sg_idx < n_sg and tts_idx < n_tts:
            cur_tt = tts[tts_idx]
            sgwc = sg_only_anc_state[sg_idx]
            sg_tt = TargetTract(sgwc.min_deact_target, sgwc.min_target, sgwc.max_target, sgwc.max_deact_target)

            if cur_tt.max_deact_target < sg_tt.min_deact_target:
                tts_idx += 1
                continue
            elif sg_tt.max_deact_target < cur_tt.min_deact_target:
                sg_idx += 1
                continue

            # Overlapping now
            if sg_tt == cur_tt:
               matching_sgs.append(sgwc.get_singleton())
            sg_idx += 1
            tts_idx += 1

        return matching_sgs

    def _get_cond_prob_singleton_del(self, singleton: Singleton):
        """
        @return the conditional probability of the deletion for the singleton happening
                (given that the target tract occurred)
        """
        left_trim_len = self.bcode_meta.abs_cut_sites[singleton.min_target] - singleton.start_pos
        right_trim_len = singleton.del_end - self.bcode_meta.abs_cut_sites[singleton.max_target] + 1
        left_trim_long_min = self.bcode_meta.left_long_trim_min[singleton.min_target]
        right_trim_long_min = self.bcode_meta.right_long_trim_min[singleton.max_target]
        if singleton.is_left_long:
            min_left_trim = left_trim_long_min
            max_left_trim = self.bcode_meta.left_max_trim[singleton.min_target]
        else:
            min_left_trim = 0
            max_left_trim = left_trim_long_min - 1

        if singleton.is_right_long:
            min_right_trim = right_trim_long_min
            max_right_trim = self.bcode_meta.right_max_trim[singleton.min_target]
        else:
            min_right_trim = 0
            max_right_trim = right_trim_long_min - 1

        left_prob = BoundedPoisson(min_left_trim, max_left_trim, self.trim_poisson_params[0]).pmf(
            left_trim_len)
        right_prob = BoundedPoisson(min_right_trim, max_right_trim, self.trim_poisson_params[1]).pmf(
            right_trim_len)

        # Check if we should add zero inflation
        if not singleton.is_left_long and not singleton.is_right_long:
            if singleton.del_len == 0:
                return self.trim_zero_prob + (1 - self.trim_zero_prob) * left_prob * right_prob
            else:
                return (1 - self.trim_zero_prob) * left_prob * right_prob
        else:
            return left_prob * right_prob

    def _get_cond_prob_singleton_insert(self, singleton: Singleton):
        """
        @return the conditional probability of the insertion for this singleton happening
                (given that the target tract occurred)
        """
        insert_len = singleton.insert_len
        insert_len_prob = poisson(self.insert_poisson_param).pmf(insert_len)
        # There are 4^insert_len insert string to choose from
        # Assuming all insert strings are equally likely
        insert_seq_prob = 1.0/np.power(4, insert_len)

        # Check if we should add zero inflation
        if insert_len == 0:
            return self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob
        else:
            return (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob

    @staticmethod
    def get_possible_target_tracts(active_any_targs: List[int]):
        """
        @param active_any_targs: a list of active targets that can be cut with any trim
        @return a set of possible target tracts
        """
        n_any_targs = len(active_any_targs)

        # Take one step from this TT group using two step procedure
        # 1. enumerate all possible start positions for target tract
        # 2. enumerate all possible end positions for target tract

        # List possible starts of the target tracts
        all_starts = [[] for _ in range(n_any_targs)]
        for i0_prime, t0_prime in enumerate(active_any_targs):
            # No left trim overflow
            all_starts[i0_prime].append((t0_prime, t0_prime))
            # Determine if left trim overflow allowed
            if i0_prime < n_any_targs - 1 and active_any_targs[i0_prime + 1] == t0_prime + 1:
                # Add in the long left trim overflow
                all_starts[i0_prime + 1].append((t0_prime, t0_prime + 1))

        # Create possible ends of the target tracts
        all_ends = [[] for i in range(n_any_targs)]
        for i1_prime, t1_prime in enumerate(active_any_targs):
            # No right trim overflow
            all_ends[i1_prime].append((t1_prime, t1_prime))
            # Determine if right trim overflow allowed
            if i1_prime > 0 and active_any_targs[i1_prime - 1] == t1_prime - 1:
                # Add in the right trim overflow
                all_ends[i1_prime - 1].append((t1_prime - 1, t1_prime))

        # Finally create all possible target tracts by combining possible start and ends
        tt_evts = set()
        for j, tt_starts in enumerate(all_starts):
            for k in range(j, n_any_targs):
                tt_ends = all_ends[k]
                for tt_start in tt_starts:
                    for tt_end in tt_ends:
                        tt_evt = TargetTract(tt_start[0], tt_start[1], tt_end[0], tt_end[1])
                        tt_evts.add(tt_evt)

        return tt_evts

