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
from constants import UNLIKELY
from bounded_poisson import BoundedPoisson
from test import myexpm

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
        self._create_placeholders()
        self._create_hazard()
        self._create_hazard_away()
        self._create_del_probs()
        self._create_insert_probs()
        self._create_indel_probs()

    def _create_placeholders(self):
        # placeholders for info about an indel
        self.targets_ph = tf.placeholder(tf.int32, [None, 2])
        self.min_target = self.targets_ph[:,0]
        self.max_target = self.targets_ph[:,1]
        self.long_status_ph = tf.placeholder(tf.float32, [None, 2])
        self.is_left_long = self.long_status_ph[:,0]
        self.is_right_long = self.long_status_ph[:,1]
        self.positions_ph = tf.placeholder(tf.int32, [None, 3])
        self.start_pos = self.positions_ph[:,0]
        self.del_len = self.positions_ph[:,1]
        self.insert_len = tf.cast(self.positions_ph[:,2], dtype=tf.float32)
        self.del_end = self.start_pos + self.del_len

        # placeholders for info about target status -- used by hazard away calculation
        self.left_trimmables_ph = tf.placeholder(tf.int32, [None, self.num_targets])
        self.right_trimmables_ph = tf.placeholder(tf.int32, [None, self.num_targets])

    def _create_hazard(self):
        """
        Helpers for creating hazard in tensorflow graph and its associated gradient
        """
        # Compute the hazard
        log_lambda_part = tf.log(tf.gather(self.target_lams, self.min_target)) + tf.log(tf.gather(self.target_lams, self.max_target)) * tf_common.not_equal_float(self.min_target, self.max_target)
        left_trim_prob = tf_common.ifelse(self.is_left_long, self.trim_long_probs[0], 1 - self.trim_long_probs[0])
        right_trim_prob = tf_common.ifelse(self.is_right_long, self.trim_long_probs[1], 1 - self.trim_long_probs[1])
        self.hazard = tf.exp(log_lambda_part + tf.log(left_trim_prob) + tf.log(right_trim_prob))

        # Compute the gradient
        # TODO: This does not do the right thing!!!
        self.hazard_grad = self.grad_opt.compute_gradients(
            self.hazard,
            var_list=[self.all_vars])

    def _create_hazard_away(self):
        """
        Helpers for creating hazard-away in tensorflow graph and its associated gradient
        """
        # Compute the hazard away
        focal_hazards = self._create_hazard_list(True, True, self.left_trimmables_ph, self.right_trimmables_ph)
        left_hazards = self._create_hazard_list(True, False, self.left_trimmables_ph, self.right_trimmables_ph)[:, :self.num_targets - 1]
        right_hazards = self._create_hazard_list(False, True, self.left_trimmables_ph, self.right_trimmables_ph)[:, 1:]
        left_cum_hazards = tf.cumsum(left_hazards, axis=1)
        inter_target_hazards = tf.multiply(left_cum_hazards, right_hazards)
        self.hazard_away = tf.reduce_sum(focal_hazards, axis=1) + tf.reduce_sum(inter_target_hazards, axis=1)

        # Compute the gradient
        # TODO: This does not do the right thing!!!
        self.hazard_away_grad = self.grad_opt.compute_gradients(
            self.hazard_away,
            var_list=[self.all_vars])

    def _create_hazard_list(self, trim_left: bool, trim_right: bool, left_trimmables, right_trimmables):
        """
        Helper function for creating hazard list nodes -- useful for calculating hazard away
        """
        trim_short_left = 1 - self.trim_long_probs[0] if trim_left else 1
        trim_short_right = 1 - self.trim_long_probs[1] if trim_right else 1

        left_factor = tf_common.equal_float(left_trimmables, 1) * trim_short_left + tf_common.equal_float(left_trimmables, 2)

        right_factor = tf_common.equal_float(self.right_trimmables_ph, 1) * trim_short_right + tf_common.equal_float(self.right_trimmables_ph, 2)

        hazard_list = self.target_lams * left_factor * right_factor
        return hazard_list

    def _create_del_probs(self):
        # Compute conditional prob of deletion for a singleton
        left_trim_len = tf.cast(tf.gather(self.bcode_meta.abs_cut_sites, self.min_target) - self.start_pos, tf.float32)
        right_trim_len = tf.cast(self.del_end - tf.gather(self.bcode_meta.abs_cut_sites, self.max_target), tf.float32)
        left_trim_long_min = tf.cast(tf.gather(self.bcode_meta.left_long_trim_min, self.min_target), tf.float32)
        right_trim_long_min = tf.cast(tf.gather(self.bcode_meta.right_long_trim_min, self.max_target), tf.float32)

        min_left_trim = self.is_left_long * left_trim_long_min
        max_left_trim = tf_common.ifelse(
                self.is_left_long,
                tf.cast(tf.gather(self.bcode_meta.left_max_trim, self.min_target), tf.float32),
                left_trim_long_min - 1)
        min_right_trim = self.is_right_long * right_trim_long_min
        max_right_trim = tf_common.ifelse(
                self.is_right_long,
                tf.cast(tf.gather(self.bcode_meta.right_max_trim, self.max_target), tf.float32),
                right_trim_long_min - 1)

        # TODO: using a uniform distribution for now
        check_left_max = tf.cast(tf.less_equal(left_trim_len, max_left_trim), tf.float32)
        check_left_min = tf.cast(tf.less_equal(min_left_trim, left_trim_len), tf.float32)
        #TODO: check this range thing
        left_prob = 1.0/(max_left_trim - min_left_trim + 1) * check_left_max #* check_left_min
        #left_prob = tf_distributions.Normal(self.trim_poissons[0], self.trim_poissons[1]).cdf(0.5)
        check_right_max = tf.cast(tf.less_equal(right_trim_len, max_right_trim), tf.float32)
        check_right_min = tf.cast(tf.less_equal(min_right_trim, right_trim_len), tf.float32)
        right_prob = 1.0/(max_right_trim - min_right_trim + 1) * check_right_max * check_right_min

        is_short_indel = tf_common.equal_float(self.is_left_long + self.is_right_long, 0)
        is_len_zero = tf_common.equal_float(self.del_len, 0)
        self.del_prob = tf_common.ifelse(is_short_indel,
                tf_common.ifelse(is_len_zero,
                    self.trim_zero_prob + (1 - self.trim_zero_prob) * left_prob * right_prob,
                    (1 - self.trim_zero_prob) * left_prob * right_prob),
                left_prob * right_prob)

    def _create_insert_probs(self):
        poiss_unstd = tf.exp(-self.insert_poisson) * tf.pow(self.insert_poisson, self.insert_len)
        insert_len_prob = poiss_unstd/tf.exp(tf.lgamma(self.insert_len + 1))
        insert_seq_prob = 1.0/tf.pow(4.0, self.insert_len)
        is_insert_zero = tf.cast(tf.equal(self.insert_len, 0), dtype=tf.float32)
        self.insert_prob = tf_common.ifelse(
                is_insert_zero,
                self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob,
                (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob)

    def _create_indel_probs(self):
        self.singleton_cond_prob = self.insert_prob * self.del_prob
        self.singleton_cond_grad = self.grad_opt.compute_gradients(
                self.singleton_cond_prob,
                var_list = [self.all_vars])

    def create_topology_log_lik(self, transition_matrix_wrappers: Dict):
        """
        This is total bogus -- just want to see how to create a lot of nodes
        """
        self.log_lik = 0
        self.L = dict() # Stores normalized probs
        self.pt_matrix = dict()
        self.trim_probs = dict()
        for node in self.topology.traverse("postorder"):
            if node.is_leaf():
                trans_mat_w = transition_matrix_wrappers[node.node_id]
                self.L[node.node_id] = np.zeros((trans_mat_w.num_states, 1))
                assert len(node.state_sum.tts_list) == 1
                tts_key = trans_mat_w.key_dict[node.state_sum.tts_list[0]]
                self.L[node.node_id][tts_key] = 1
                # Convert to tensorflow usage
                self.L[node.node_id] = tf.constant(self.L[node.node_id], dtype=tf.float32)
            else:
                self.L[node.node_id] = 1.0
                for child in node.children:
                    ch_trans_mat_w = transition_matrix_wrappers[child.node_id]
                    trans_mat = self._init_transition_matrix(ch_trans_mat_w)
                    # Get the trim probabilities
                    # TODO: implement later
                    #trim_probs[child.node_id] = self._get_trim_probs(
                    #        ch_trans_mat,
                    #        unmasked_singleton_probs,
                    #        node,
                    #        child)

                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = self.branch_lens[child.node_id]
                    pr_matrix, _, _, _ = myexpm(trans_mat, branch_len)
                    self.pt_matrix[child.node_id] = pr_matrix

                    # Get the probability for the data descended from the child node, assuming that the node
                    # has a particular target tract repr.
                    # These down probs are ordered according to the child node's numbering of the TTs states
                    # TODO: add in trim probabilities
                    ch_ordered_down_probs = tf.matmul(
                            self.pt_matrix[child.node_id],
                            self.L[child.node_id])

                    if not node.is_root():
                        # Reorder summands according to node's numbering of tts states
                        trans_mat_w = transition_matrix_wrappers[node.node_id]

                        indices = [trans_mat_w.key_dict[tts] for tts in node.state_sum.tts_list]
                        vals = [ch_ordered_down_probs[ch_trans_mat_w.key_dict[tts]] for tts in node.state_sum.tts_list]
                        down_probs = tf.sparse_to_dense(
                                indices,
                                [trans_mat_w.num_states,1],
                                vals)

                        self.L[node.node_id] *= down_probs
                    else:
                        # For the root node, we just want the probability where the root node is unmodified
                        # No need to reorder
                        ch_id = ch_trans_mat_w.key_dict[()]
                        self.L[node.node_id] *= ch_ordered_down_probs[ch_id]

        self.log_lik = tf.log(self.L[self.root_node_id])

    def _init_transition_matrix(self, matrix_wrapper: TransitionMatrixWrapper):
        unlikely_key = matrix_wrapper.num_states

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

        # Now fill in the matrix
        idx = 0
        matrix_coo = []
        matrix_vals = []
        for i, (start_tts, matrix_row) in enumerate(matrix_wrapper.matrix_dict.items()):
            start_key = matrix_wrapper.key_dict[start_tts]
            # Tracks the total hazard to the likely states
            haz_to_likely = 0
            for end_tts, tt_evt in matrix_row.items():
                haz = hazard_nodes[idx]
                end_key = matrix_wrapper.key_dict[end_tts]
                matrix_coo.append([start_key, end_key])
                matrix_vals.append(haz)
                haz_to_likely += haz
                idx += 1

            haz_away = hazard_away_nodes[i]
            # Hazard to unlikely state is hazard away minus hazard to likely states
            matrix_coo.append([start_key,unlikely_key])
            matrix_vals.append(haz_away - haz_to_likely)
            # Hazard of staying is negative of hazard away
            matrix_coo.append([start_key,unlikely_key])
            matrix_vals.append(-haz_away)

        matrix_vals = tf.stack(matrix_vals, axis=0)
        q_matrix = tf.sparse_to_dense(
                sparse_indices=matrix_coo,
                output_shape=[matrix_wrapper.num_states + 1, matrix_wrapper.num_states + 1],
                sparse_values=matrix_vals)
        return q_matrix

    def initialize_transition_matrices(self, transition_mat_wrappers: Dict[int, TransitionMatrixWrapper]):
        """
        @param transition_mat_wrappers: the list of transition matrix wrappers, which are just skeletons for
                                        the actual matrix

        @return a list of real transition matrices based on this model's parameters
        """
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
            # TODO: maybe combine them to be more efficient?
            hazard_aways, hazard_away_grads = self.get_hazard_aways(start_tts_list)
            hazards, hazard_grads = self.get_hazards(tt_evts)

            # Now fill in the matrix
            idx = 0
            matrix_dict = dict()
            matrix_grad_dict = dict()
            for i, (start_tts, matrix_row) in enumerate(matrix_wrapper.matrix_dict.items()):
                matrix_dict[start_tts] = dict()
                matrix_grad_dict[start_tts] = dict()
                # Tracks the total hazard to the likely states
                haz_to_likely = 0
                haz_grad_to_likely = 0
                for end_tts, tt_evt in matrix_row.items():
                    haz = hazards[idx]
                    haz_grad = hazard_grads[idx]
                    print("HAZ GRAD", haz_grad)
                    matrix_dict[start_tts][end_tts] = haz
                    matrix_grad_dict[start_tts][end_tts] = haz_grad
                    haz_to_likely += haz
                    haz_grad_to_likely += haz_grad
                    idx += 1

                haz_away = hazard_aways[i]
                haz_away_grad = hazard_away_grads[i]
                # Hazard to unlikely state is hazard away minus hazard to likely states
                matrix_dict[start_tts][UNLIKELY] = haz_away - haz_to_likely
                matrix_grad_dict[start_tts][UNLIKELY] = haz_away_grad - haz_grad_to_likely
                # Hazard of staying is negative of hazard away
                matrix_dict[start_tts][start_tts] = -haz_away
                matrix_grad_dict[start_tts][start_tts] = -haz_away_grad

            # Add unlikely state
            matrix_dict[UNLIKELY] = dict()
            matrix_grad_dict[UNLIKELY] = dict()

            all_matrices[node_id] = TransitionMatrix(matrix_dict, matrix_grad_dict)
        return all_matrices

    def get_hazards(self, tt_evts: List[TargetTract]):
        """
        Propagates through the tensorflow graph to obtain hazard of each TargetTract in `tt_evts`
        """
        if len(tt_evts) == 0:
            return [], []

        target_inputs = []
        long_status_inputs = []
        for tt_evt in tt_evts:
            target_inputs.append([tt_evt.min_target, tt_evt.max_target])
            long_status_inputs.append([tt_evt.is_left_long, tt_evt.is_right_long])

        hazards, hazard_grads = self.sess.run(
                [self.hazard, self.hazard_grad],
                feed_dict={
                    self.targets_ph: target_inputs,
                    self.long_status_ph: long_status_inputs})
        hazard_grads = [g_tup[0] for g_tup in hazard_grads]
        return hazards, hazard_grads

    def get_hazard(self, tt_evt: TargetTract):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        hazard, _ = self.get_hazards([tt_evt])
        return hazard[0]

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
        target_inputs = []
        long_status_inputs = []
        for tt_evt in tt_evts:
            target_inputs.append([tt_evt.min_target, tt_evt.max_target])
            long_status_inputs.append([tt_evt.is_left_long, tt_evt.is_right_long])

        # TODO: actually do something


    def _create_hazard_away_nodes(self, tts_list: List[Tuple[TargetTract]]):
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
        hazard_away = tf.reduce_sum(focal_hazards, axis=1) + tf.reduce_sum(inter_target_hazards, axis=1)
        return hazard_away


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

        hazard_aways, hazard_aways_grad = self.sess.run(
                [self.hazard_away, self.hazard_away_grad],
                feed_dict = {
                    self.left_trimmables_ph: left_trimmables,
                    self.right_trimmables_ph: right_trimmables})
        hazard_aways_grad = [g_tup[0] for g_tup in hazard_aways_grad]
        return hazard_aways, hazard_aways_grad

    def get_hazard_away(self, tts: Tuple[TargetTract]):
        hazard_away, _ = self.get_hazard_aways([tts])
        return hazard_away[0]

    def initialize_indel_cond_probs(self, singletons: List[Singleton]):
        """
        @param singletons: a list of singletons that we want the conditional probability

        @return a dict mapping a singleton to its conditional probability
        """
        if len(singletons) == 0:
            return dict()

        targets = []
        long_statuses = []
        positions = []
        for sg in singletons:
            targets.append([sg.min_target, sg.max_target])
            long_statuses.append([sg.is_left_long, sg.is_right_long])
            positions.append([sg.start_pos, sg.del_len, sg.insert_len])

        singleton_cond_prob = self.sess.run(self.singleton_cond_prob, feed_dict={
            self.targets_ph: targets,
            self.long_status_ph: long_statuses,
            self.positions_ph: positions})

        singleton_prob_dict = dict()
        for i, sg in enumerate(singletons):
            singleton_prob_dict[sg] = singleton_cond_prob[i]
        return singleton_prob_dict

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
