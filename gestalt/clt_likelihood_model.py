import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tf_distributions

from typing import List, Tuple, Dict
from numpy import ndarray
from scipy.stats import poisson
from tensorflow import Session

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract, AncState, SingletonWC, Singleton
from transition_matrix import TransitionMatrixWrapper, TransitionMatrix
from common import merge_target_tract_groups
import tf_common
from common import target_tract_repr_diff
from constants import UNLIKELY
from bounded_poisson import BoundedPoisson

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type
    """
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
            trim_poissons: ndarray = 0.9 * np.ones(2),
            insert_zero_prob: float = 0.5,
            insert_poisson: float = 2.0):
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

        self.all_vars = tf.Variable(
                np.concatenate([
                    branch_lens,
                    target_lams,
                    trim_long_probs,
                    [trim_zero_prob],
                    trim_poissons,
                    [insert_zero_prob],
                    [insert_poisson]]),
                dtype=tf.float32)

        self.branch_lens = self.all_vars[:branch_lens.size]
        prev_size = branch_lens.size
        up_to_size = branch_lens.size + target_lams.size
        self.target_lams = self.all_vars[prev_size:up_to_size]
        prev_size = up_to_size
        up_to_size += trim_long_probs.size
        self.trim_long_probs = self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        up_to_size += 1
        self.trim_zero_prob = self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        up_to_size += trim_poissons.size
        self.trim_poissons =self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        up_to_size += 1
        self.insert_zero_prob =self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        up_to_size += 1
        self.insert_poisson =self.all_vars[prev_size: up_to_size]

        self.grad_opt = tf.train.GradientDescentOptimizer(learning_rate=1)
        self._create_hazard_node_for_simulation()

    def _create_hazard_node_for_simulation(self):
        """
        creates nodes just for calculating the hazard when simulating stuff
        """
        self.targets_ph = tf.placeholder(tf.int32, [None, 2])
        self.long_status_ph = tf.placeholder(tf.float32, [None, 2])

        self.hazard = self._create_hazard(
                self.targets_ph[:,0],
                self.targets_ph[:,1],
                self.long_status_ph[:,0],
                self.long_status_ph[:,1])

    def _create_hazard(self, min_target, max_target, long_left_statuses, long_right_statuses):
        """
        Helpers for creating hazard in tensorflow graph and its associated gradient
        """
        # Compute the hazard
        log_lambda_part = tf.log(tf.gather(self.target_lams, min_target)) + tf.log(tf.gather(self.target_lams, max_target)) * tf_common.not_equal_float(min_target, max_target)
        left_trim_prob = tf_common.ifelse(long_left_statuses, self.trim_long_probs[0], 1 - self.trim_long_probs[0])
        right_trim_prob = tf_common.ifelse(long_right_statuses, self.trim_long_probs[1], 1 - self.trim_long_probs[1])
        hazard = tf.exp(log_lambda_part + tf.log(left_trim_prob) + tf.log(right_trim_prob), name="hazard")
        return hazard

    def _create_hazard_list(self, trim_left: bool, trim_right: bool, left_trimmables, right_trimmables):
        """
        Helper function for creating hazard list nodes -- useful for calculating hazard away
        """
        trim_short_left = 1 - self.trim_long_probs[0] if trim_left else 1
        trim_short_right = 1 - self.trim_long_probs[1] if trim_right else 1

        left_factor = tf_common.equal_float(left_trimmables, 1) * trim_short_left + tf_common.equal_float(left_trimmables, 2)

        right_factor = tf_common.equal_float(right_trimmables, 1) * trim_short_right + tf_common.equal_float(right_trimmables, 2)

        hazard_list = self.target_lams * left_factor * right_factor
        return hazard_list

    def _create_del_probs(self, singletons: List[Singleton]):
        min_targets = [sg.min_target for sg in singletons]
        max_targets = [sg.max_target for sg in singletons]
        is_left_longs = tf.constant(
                [sg.is_left_long for sg in singletons], dtype=tf.float32)
        is_right_longs = tf.constant(
                [sg.is_right_long for sg in singletons], dtype=tf.float32)
        start_posns = tf.constant(
                [sg.start_pos for sg in singletons])
        del_ends = tf.constant(
                [sg.del_end for sg in singletons])
        del_len = del_ends - start_posns

        # Compute conditional prob of deletion for a singleton
        left_trim_len = tf.cast(tf.gather(self.bcode_meta.abs_cut_sites, min_targets) - start_posns, tf.float32)
        right_trim_len = tf.cast(del_ends - tf.gather(self.bcode_meta.abs_cut_sites, max_targets), tf.float32)
        left_trim_long_min = tf.cast(tf.gather(self.bcode_meta.left_long_trim_min, min_targets), tf.float32)
        right_trim_long_min = tf.cast(tf.gather(self.bcode_meta.right_long_trim_min, max_targets), tf.float32)

        min_left_trim = is_left_longs * left_trim_long_min
        max_left_trim = tf_common.ifelse(
                is_left_longs,
                tf.cast(tf.gather(self.bcode_meta.left_max_trim, min_targets), tf.float32),
                left_trim_long_min - 1)
        min_right_trim = is_right_longs * right_trim_long_min
        max_right_trim = tf_common.ifelse(
                is_right_longs,
                tf.cast(tf.gather(self.bcode_meta.right_max_trim, max_targets), tf.float32),
                right_trim_long_min - 1)

        # TODO: using a uniform distribution for now
        check_left_max = tf.cast(tf.less_equal(left_trim_len, max_left_trim), tf.float32)
        check_left_min = tf.cast(tf.less_equal(min_left_trim, left_trim_len), tf.float32)
        #TODO: check this range thing
        left_prob = 1.0/(max_left_trim - min_left_trim + 1) * check_left_max #* check_left_min
        check_right_max = tf.cast(tf.less_equal(right_trim_len, max_right_trim), tf.float32)
        check_right_min = tf.cast(tf.less_equal(min_right_trim, right_trim_len), tf.float32)
        right_prob = 1.0/(max_right_trim - min_right_trim + 1) * check_right_max * check_right_min

        is_short_indel = tf_common.equal_float(is_left_longs + is_right_longs, 0)
        is_len_zero = tf_common.equal_float(del_len, 0)
        del_prob = tf_common.ifelse(is_short_indel,
                tf_common.ifelse(is_len_zero,
                    self.trim_zero_prob + (1 - self.trim_zero_prob) * left_prob * right_prob,
                    (1 - self.trim_zero_prob) * left_prob * right_prob),
                left_prob * right_prob)
        return del_prob

    def _create_insert_probs(self, singletons: List[Singleton]):
        insert_lens = tf.constant(
                [sg.insert_len for sg in singletons], dtype=tf.float32)
        poiss_unstd = tf.exp(-self.insert_poisson) * tf.pow(self.insert_poisson, insert_lens)
        insert_len_prob = poiss_unstd/tf.exp(tf.lgamma(insert_lens + 1))
        insert_seq_prob = 1.0/tf.pow(4.0, insert_lens)
        is_insert_zero = tf.cast(tf.equal(insert_lens, 0), dtype=tf.float32)
        insert_prob = tf_common.ifelse(
                is_insert_zero,
                self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob,
                (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob)
        return insert_prob

    def get_log_lik(self):
        log_lik, log_lik_grad = self.sess.run([self.log_lik, self.log_lik_grad])
        #log_lik = self.sess.run(self.log_lik)
        return log_lik, log_lik_grad

    def _create_indel_probs(self, singletons: List[Singleton]):
        """
        Create tensorflow objects for the cond prob of indels

        @return a tensorflow object with indel probs for each singleton
        """
        if not singletons:
            return []
        else:
            insert_probs = self._create_insert_probs(singletons)
            del_probs = self._create_del_probs(singletons)
            indel_probs = insert_probs * del_probs
            return indel_probs

    def create_topology_log_lik(self, transition_matrix_wrappers: Dict):
        """
        Create a tensorflow graph of the likelihood calculation
        """
        self.log_lik = 0

        singletons = CLTLikelihoodModel._get_unmasked_indels(self.topology)
        singleton_index_dict = {sg: int(i) for i, sg in enumerate(singletons)}
        singleton_cond_prob = self._create_indel_probs(singletons)

        # Store the tensorflow objects that calculate the prob of a node being in each state given the leaves
        self.L = dict()
        self.pt_matrix = dict()
        self.trim_probs = dict()
        for node in self.topology.traverse("postorder"):
            if node.is_leaf():
                trans_mat_w = transition_matrix_wrappers[node.node_id]
                self.L[node.node_id] = np.zeros((trans_mat_w.num_likely_states + 1, 1))
                assert len(node.state_sum.tts_list) == 1
                tts_key = trans_mat_w.key_dict[node.state_sum.tts_list[0]]
                self.L[node.node_id][tts_key] = 1
                # Convert to tensorflow usage
                self.L[node.node_id] = tf.constant(self.L[node.node_id], dtype=tf.float32)
            else:
                self.L[node.node_id] = 1.0
                for child in node.children:
                    ch_trans_mat_w = transition_matrix_wrappers[child.node_id]
                    trans_mat = self._create_transition_matrix(ch_trans_mat_w)
                    # Get the trim probabilities
                    self.trim_probs[child.node_id] = self._create_trim_prob_matrix(
                            ch_trans_mat_w,
                            singleton_cond_prob,
                            singleton_index_dict,
                            node,
                            child)

                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = self.branch_lens[child.node_id]
                    pr_matrix, _, _, _ = tf_common.myexpm(trans_mat, branch_len)
                    self.pt_matrix[child.node_id] = pr_matrix

                    # Get the probability for the data descended from the child node, assuming that the node
                    # has a particular target tract repr.
                    # These down probs are ordered according to the child node's numbering of the TTs states
                    ch_ordered_down_probs = tf.matmul(
                            tf.multiply(self.pt_matrix[child.node_id], self.trim_probs[child.node_id]),
                            self.L[child.node_id])

                    if not node.is_root():
                        # Reorder summands according to node's numbering of tts states
                        trans_mat_w = transition_matrix_wrappers[node.node_id]

                        down_probs = CLTLikelihoodModel._reorder_likelihoods(
                                ch_ordered_down_probs,
                                node.state_sum.tts_list,
                                trans_mat_w,
                                ch_trans_mat_w)

                        self.L[node.node_id] *= down_probs
                    else:
                        # For the root node, we just want the probability where the root node is unmodified
                        # No need to reorder
                        ch_id = ch_trans_mat_w.key_dict[()]
                        self.L[node.node_id] *= ch_ordered_down_probs[ch_id]

                scaler = tf.reduce_sum(self.L[node.node_id])
                if scaler == 0:
                    raise ValueError("Why is everything zero?")
                self.L[node.node_id] /= scaler
                self.log_lik += tf.log(scaler)

        self.log_lik += tf.log(self.L[self.root_node_id])
        self.log_lik_grad = self.grad_opt.compute_gradients(
            self.log_lik,
            var_list=[self.all_vars])

    def _create_transition_matrix(self, matrix_wrapper: TransitionMatrixWrapper):
        """
        Uses tensorflow to create the instantaneous transition matrix
        """
        unlikely_key = matrix_wrapper.num_likely_states

        # Get inputs ready for tensorflow
        hazard_list = []
        tt_evts = []
        start_tts_list = []
        for start_tts, matrix_row in matrix_wrapper.matrix_dict.items():
            start_tts_list.append(start_tts)
            for end_tts, tt_evt in matrix_row.items():
                tt_evts.append(tt_evt)

        # Gets hazards (by tensorflow)
        hazard_away_nodes = self._create_hazard_away_nodes(start_tts_list)
        hazard_nodes = self._create_hazard_nodes(tt_evts)

        # Now fill in the matrix -- match tensorflow object with indices of instant transition matrix
        idx = 0
        index_vals = []
        for i, (start_tts, matrix_row) in enumerate(matrix_wrapper.matrix_dict.items()):
            start_key = matrix_wrapper.key_dict[start_tts]
            haz_away = hazard_away_nodes[i]

            # Hazard of staying is negative of hazard away
            index_vals.append([[start_key, start_key], -haz_away])

            # Tracks the total hazard to the likely states
            haz_to_likely = 0
            for end_tts, tt_evt in matrix_row.items():
                haz = hazard_nodes[idx]
                end_key = matrix_wrapper.key_dict[end_tts]
                index_vals.append([[start_key, end_key], haz])
                haz_to_likely += haz
                idx += 1

            # Hazard to unlikely state is hazard away minus hazard to likely states
            index_vals.append([[start_key, unlikely_key], haz_away - haz_to_likely])

        q_matrix = tf_common.sparse_to_dense(
                index_vals,
                output_shape=[matrix_wrapper.num_likely_states + 1, matrix_wrapper.num_likely_states + 1],
                name="top.q_matrix")
        return q_matrix

    def _create_trim_prob_matrix(self,
            ch_trans_mat_w: TransitionMatrixWrapper,
            singleton_cond_prob, # Tensorflow array
            singleton_index_dict: Dict[int, Singleton],
            node: CellLineageTree,
            child: CellLineageTree):
        """
        @param model: model parameter
        @param ch_trans_mat: the transition matrix corresponding to child node (we make sure the entries in the trim prob matrix match
                        the order in ch_trans_mat)
        @param node: the parent node
        @param child: the child node

        @return matrix of conditional probabilities of each trim
        """
        index_vals = []

        for node_tts in node.state_sum.tts_list:
            node_tts_key = ch_trans_mat_w.key_dict[node_tts]
            for child_tts in child.state_sum.tts_list:
                child_tts_key = ch_trans_mat_w.key_dict[child_tts]

                diff_target_tracts = target_tract_repr_diff(node_tts, child_tts)
                singletons = CLTLikelihoodModel.get_matching_singletons(child.anc_state, diff_target_tracts)

                if singletons:
                    trim_probs = tf.gather(
                            params = singleton_cond_prob,
                            indices = [singleton_index_dict[sg] for sg in singletons])
                    val = tf.reduce_prod(trim_probs)
                    index_vals.append([[node_tts_key, child_tts_key], val])

        output_shape = [ch_trans_mat_w.num_likely_states + 1, ch_trans_mat_w.num_likely_states + 1]
        if index_vals:
            return tf_common.sparse_to_dense(
                index_vals,
                output_shape,
                default_value = 1,
                name="top.trim_probs")
        else:
            return tf.ones(output_shape)

    def get_hazard(self, tt_evt: TargetTract):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        hazards = self.sess.run(
                self.hazard,
                feed_dict={
                    self.targets_ph: [[tt_evt.min_target, tt_evt.max_target]],
                    self.long_status_ph: [[tt_evt.is_left_long, tt_evt.is_right_long]]})
        return hazards[0]

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

    def _create_hazard_nodes(self, tt_evts: List[TargetTract]):
        """
        @return tensorfow array of the hazard of introducing each target tract in `tt_evts`
        """
        min_targets = tf.constant([tt_evt.min_target for tt_evt in tt_evts], dtype=tf.int32)
        max_targets = tf.constant([tt_evt.max_target for tt_evt in tt_evts], dtype=tf.int32)
        long_left_statuses = tf.constant([tt_evt.is_left_long for tt_evt in tt_evts], dtype=tf.float32)
        long_right_statuses = tf.constant([tt_evt.is_right_long for tt_evt in tt_evts], dtype=tf.float32)

        # Compute the hazard
        hazard_nodes = self._create_hazard(min_targets, max_targets, long_left_statuses, long_right_statuses)
        return hazard_nodes

    def get_hazard_away(self, tts: Tuple[TargetTract]):
        haz_away = self._create_hazard_away_nodes([tts])
        return self.sess.run(haz_away)[0]

    def _create_hazard_away_nodes(self, tts_list: List[Tuple[TargetTract]]):
        """
        @return tensorfow array of the hazard away from each target tract repr in `tts_list`
        """
        left_trimmables = []
        right_trimmables = []
        for tts in tts_list:
            left_m, right_m = self._get_hazard_masks(tts)
            left_trimmables.append(left_m)
            right_trimmables.append(right_m)

        # Compute the hazard away
        focal_hazards = self._create_hazard_list(True, True, left_trimmables, right_trimmables)
        left_hazards = self._create_hazard_list(True, False, left_trimmables, right_trimmables)[:, :self.num_targets - 1]
        right_hazards = self._create_hazard_list(False, True, left_trimmables, right_trimmables)[:, 1:]
        left_cum_hazards = tf.cumsum(left_hazards, axis=1)
        inter_target_hazards = tf.multiply(left_cum_hazards, right_hazards)
        hazard_away_nodes = tf.add(
                tf.reduce_sum(focal_hazards, axis=1),
                tf.reduce_sum(inter_target_hazards, axis=1),
                name="hazard_away")
        return hazard_away_nodes

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

    @staticmethod
    def _get_unmasked_indels(topology: CellLineageTree):
        """
        Determine the set of singletons we need to calculate the indel prob for

        @return Set[Singleton] that are unmasked (set of the values of branch_to_singletons)
        """
        singletons = set()
        for node in topology.traverse("postorder"):
            if not node.is_leaf():
                for child in node.children:
                    for node_tts in node.state_sum.tts_list:
                        for child_tts in child.state_sum.tts_list:
                            diff_target_tracts = target_tract_repr_diff(node_tts, child_tts)
                            matching_sgs = CLTLikelihoodModel.get_matching_singletons(child.anc_state, diff_target_tracts)
                            singletons.update(matching_sgs)
        return singletons

    @staticmethod
    def _reorder_likelihoods(
            ch_ordered_down_probs,
            tts_list:List[Tuple[TargetTract]],
            trans_mat_w: TransitionMatrixWrapper,
            ch_trans_mat_w: TransitionMatrixWrapper):
        """
        @param ch_ordered_down_probs: the Tensorflow array to be re-ordered
        @param tts_list: list of target tract reprs to include in the vector
                        rest can be set to zero
        @param node_trans_mat: provides the desired ordering
        @param ch_trans_mat: provides the ordering used in vec_lik

        @return the reordered version of vec_lik according to the order in node_trans_mat
        """
        index_vals = [
            [[trans_mat_w.key_dict[tts], 0], ch_ordered_down_probs[ch_trans_mat_w.key_dict[tts]]]
            for tts in tts_list]
        down_probs = tf_common.sparse_to_dense(
                index_vals,
                output_shape=[trans_mat_w.num_likely_states + 1, 1],
                name="top.down_probs")
        return down_probs

