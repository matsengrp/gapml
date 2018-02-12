import time
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
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
from common import target_tract_repr_diff, inv_sigmoid
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
    gamma_prior = (1,0.2)

    def __init__(self,
            topology: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            sess: Session,
            branch_lens: ndarray = None,
            target_lams: ndarray = None,
            trim_long_probs: ndarray = 0.05 * np.ones(2),
            trim_zero_prob: float = 0.5,
            trim_poissons: ndarray = 2.5 * np.ones(2),
            insert_zero_prob: float = 0.5,
            insert_poisson: float = 0.2):
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
        self._create_parameters(
                branch_lens,
                target_lams,
                trim_long_probs,
                [trim_zero_prob],
                trim_poissons,
                [insert_zero_prob],
                [insert_poisson])

        self.grad_opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self._create_hazard_node_for_simulation()

    def _create_parameters(self,
            branch_lens: ndarray,
            target_lams: ndarray,
            trim_long_probs: ndarray,
            trim_zero_prob: float,
            trim_poissons: ndarray,
            insert_zero_prob: float,
            insert_poisson: float):
        self.all_vars = tf.Variable(
                np.concatenate([
                    target_lams,
                    inv_sigmoid(trim_long_probs),
                    inv_sigmoid(trim_zero_prob),
                    trim_poissons,
                    inv_sigmoid(insert_zero_prob),
                    insert_poisson,
                    branch_lens]),
                dtype=tf.float64)
        self.all_vars_ph = tf.placeholder(tf.float64, shape=self.all_vars.shape)
        self.assign_all_vars = self.all_vars.assign(self.all_vars_ph)

        up_to_size = target_lams.size
        self.target_lams = self.all_vars[:up_to_size]
        prev_size = up_to_size
        up_to_size += trim_long_probs.size
        self.trim_long_probs = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += 1
        self.trim_zero_prob = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += trim_poissons.size
        self.trim_poissons = self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        up_to_size += 1
        self.insert_zero_prob = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += 1
        self.insert_poisson = self.all_vars[prev_size: up_to_size]
        self.branch_lens = self.all_vars[-branch_lens.size:]

    def get_vars(self):
        return self.sess.run([self.target_lams, self.trim_long_probs, self.trim_zero_prob, self.trim_poissons, self.insert_zero_prob, self.insert_poisson, self.branch_lens])

    def _create_hazard_node_for_simulation(self):
        """
        creates nodes just for calculating the hazard when simulating stuff
        """
        self.targets_ph = tf.placeholder(tf.int32, [None, 2])
        self.long_status_ph = tf.placeholder(tf.float64, [None, 2])

        self.hazard = self._create_hazard(
                self.targets_ph[:,0],
                self.targets_ph[:,1],
                self.long_status_ph[:,0],
                self.long_status_ph[:,1])

    def _create_hazard(self,
            min_target: List[int],
            max_target: List[int],
            long_left_statuses: List[bool],
            long_right_statuses: List[bool]):
        """
        Helpers for creating hazard in tensorflow graph and its associated gradient
        The arguments should all have the same length.
        The i-th elem in each argument corresponds to the target tract that was introduced.

        @return tensorflow tensor with the i-th value corresponding to the i-th target tract in the arguments
        """
        # Compute the hazard
        log_lambda_part = tf.log(tf.gather(self.target_lams, min_target)) + tf.log(tf.gather(self.target_lams, max_target)) * tf_common.not_equal_float(min_target, max_target)
        left_trim_prob = tf_common.ifelse(long_left_statuses, self.trim_long_probs[0], 1 - self.trim_long_probs[0])
        right_trim_prob = tf_common.ifelse(long_right_statuses, self.trim_long_probs[1], 1 - self.trim_long_probs[1])
        hazard = tf.exp(log_lambda_part + tf.log(left_trim_prob) + tf.log(right_trim_prob), name="hazard")
        return hazard

    def _create_hazard_list(self,
            trim_left: bool,
            trim_right: bool,
            left_trimmables: List[List[int]],
            right_trimmables: List[List[int]]):
        """
        Helper function for creating hazard list nodes -- useful for calculating hazard away
        The list-type arguments should all have the same length.
        The i-th elem in each argument corresponds to the same target tract repr.

        @return tensorflow tensor with the i-th value corresponding to the i-th target tract repr in the arguments
        """
        trim_short_left = 1 - self.trim_long_probs[0] if trim_left else 1
        trim_short_right = 1 - self.trim_long_probs[1] if trim_right else 1

        left_factor = tf_common.equal_float(left_trimmables, 1) * trim_short_left + tf_common.equal_float(left_trimmables, 2)
        right_factor = tf_common.equal_float(right_trimmables, 1) * trim_short_right + tf_common.equal_float(right_trimmables, 2)

        hazard_list = self.target_lams * left_factor * right_factor
        return hazard_list

    def _create_log_del_probs(self, singletons: List[Singleton]):
        """
        Creates tensorflow nodes that calculate the log conditional probability of the deletions found in
        each of the singletons

        @return List[tensorflow nodes] for each singleton in `singletons`
        """
        min_targets = [sg.min_target for sg in singletons]
        max_targets = [sg.max_target for sg in singletons]
        is_left_longs = tf.constant(
                [sg.is_left_long for sg in singletons], dtype=tf.float64)
        is_right_longs = tf.constant(
                [sg.is_right_long for sg in singletons], dtype=tf.float64)
        start_posns = tf.constant(
                [sg.start_pos for sg in singletons], dtype=tf.float64)
        del_ends = tf.constant(
                [sg.del_end for sg in singletons], dtype=tf.float64)
        del_len = del_ends - start_posns

        # Compute conditional prob of deletion for a singleton
        min_target_sites = tf.constant([self.bcode_meta.abs_cut_sites[mt] for mt in min_targets], dtype=tf.float64)
        max_target_sites = tf.constant([self.bcode_meta.abs_cut_sites[mt] for mt in max_targets], dtype=tf.float64)
        left_trim_len = min_target_sites - start_posns
        right_trim_len = del_ends - max_target_sites

        left_trim_long_min = tf.constant([self.bcode_meta.left_long_trim_min[mt] for mt in max_targets], dtype=tf.float64)
        right_trim_long_min = tf.constant([self.bcode_meta.right_long_trim_min[mt] for mt in max_targets], dtype=tf.float64)
        left_trim_long_max = tf.constant([self.bcode_meta.left_max_trim[mt] for mt in max_targets], dtype=tf.float64)
        right_trim_long_max = tf.constant([self.bcode_meta.right_max_trim[mt] for mt in max_targets], dtype=tf.float64)

        min_left_trim = is_left_longs * left_trim_long_min
        max_left_trim = tf_common.ifelse(is_left_longs, left_trim_long_max, left_trim_long_min - 1)
        min_right_trim = is_right_longs * right_trim_long_min
        max_right_trim = tf_common.ifelse(is_right_longs, right_trim_long_max, right_trim_long_min - 1)

        # TODO: using a uniform distribution for now
        check_left_max = tf.cast(tf.less_equal(left_trim_len, max_left_trim), tf.float64)
        check_left_min = tf.cast(tf.less_equal(min_left_trim, left_trim_len), tf.float64)
        left_prob = 1.0/(max_left_trim - min_left_trim + 1.0) * check_left_max * check_left_min
        check_right_max = tf.cast(tf.less_equal(right_trim_len, max_right_trim), tf.float64)
        check_right_min = tf.cast(tf.less_equal(min_right_trim, right_trim_len), tf.float64)
        right_prob = 1.0/(max_right_trim - min_right_trim + 1.0) * check_right_max * check_right_min

        lr_prob = left_prob * right_prob
        is_short_indel = tf_common.equal_float(is_left_longs + is_right_longs, 0)
        is_len_zero = tf_common.equal_float(del_len, 0)
        del_prob = tf_common.ifelse(is_short_indel,
                tf_common.ifelse(is_len_zero,
                    self.trim_zero_prob + (1.0 - self.trim_zero_prob) * lr_prob,
                    (1.0 - self.trim_zero_prob) * lr_prob),
                lr_prob)
        return tf.log(del_prob)

    def _create_log_insert_probs(self, singletons: List[Singleton]):
        """
        Creates tensorflow nodes that calculate the log conditional probability of the insertions found in
        each of the singletons

        @return List[tensorflow nodes] for each singleton in `singletons`
        """
        insert_lens = tf.constant(
                [sg.insert_len for sg in singletons], dtype=tf.float64)
        poiss_unstd = tf.exp(-self.insert_poisson) * tf.pow(self.insert_poisson, insert_lens)
        insert_len_prob = poiss_unstd/tf.exp(tf.lgamma(insert_lens + 1))
        insert_seq_prob = 1.0/tf.pow(tf.constant(4.0, dtype=tf.float64), insert_lens)
        is_insert_zero = tf.cast(tf.equal(insert_lens, 0), dtype=tf.float64)
        insert_prob = tf_common.ifelse(
                is_insert_zero,
                self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob,
                (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob)
        return tf.log(insert_prob)

    def _create_log_indel_probs(self, singletons: List[Singleton]):
        """
        Create tensorflow objects for the cond prob of indels

        @return list of tensorflow tensors with indel probs for each singleton
        """
        if not singletons:
            return []
        else:
            log_insert_probs = self._create_log_insert_probs(singletons)
            log_del_probs = self._create_log_del_probs(singletons)
            log_indel_probs = log_del_probs + log_insert_probs
            return log_indel_probs

    def _create_hazard_dict(self, transition_matrix_wrappers: List[TransitionMatrixWrapper]):
        """
        @param transition_matrix_wrappers: iterable with transition matrix wrappers

        @return Dict mapping the target tract introduced to its tensorflow tensor
                a tensorflow tensor with the calculations for the hazard of introducing the target tracts
        """
        tt_evts = set()
        for trans_mat_wrapper in transition_matrix_wrappers:
            # Get inputs ready for tensorflow
            for start_tts, matrix_row in trans_mat_wrapper.matrix_dict.items():
                tt_evts.update(matrix_row.values())

        # Gets hazards (by tensorflow)
        tt_evts = list(tt_evts)
        hazard_evt_dict = {tt_evt: int(i) for i, tt_evt in enumerate(tt_evts)}
        hazard_nodes = self._create_hazard_nodes(tt_evts)
        return hazard_evt_dict, hazard_nodes

    def _create_hazard_away_dict(self, transition_matrix_wrappers: List[TransitionMatrixWrapper]):
        """
        @param transition_matrix_wrappers: iterable with transition matrix wrappers

        @return Dict mapping the tuple of start target tract repr to a hazard away node
                a tensorflow tensor with the calculations for the hazard away
        """
        tts_starts = set()
        for trans_mat_wrapper in transition_matrix_wrappers.values():
            # Get inputs ready for tensorflow
            for start_tts, matrix_row in trans_mat_wrapper.matrix_dict.items():
                tts_starts.add(start_tts)

        # Gets hazards (by tensorflow)
        tts_starts = list(tts_starts)
        hazard_away_dict = {tts: int(i) for i, tts in enumerate(tts_starts)}
        hazard_away_nodes = self._create_hazard_away_nodes(tts_starts)
        return hazard_away_dict, hazard_away_nodes

    def get_log_lik(self, get_grad=False, do_logging=False):
        """
        @return the log likelihood and the gradient, if requested
        """
        if get_grad and not do_logging:
            log_lik, grad = self.sess.run(
                    [self.log_lik, self.log_lik_grad])
            return log_lik, grad[0][0]
        elif get_grad and do_logging:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            log_lik, grad, D_vals = self.sess.run(
                    [self.log_lik, self.log_lik_grad, list(self.D.values())],
                    options=run_options,
                    run_metadata=run_metadata)

            self.profile_writer.add_run_metadata(run_metadata, "hello?")

            # Quick check that all the diagonal matrix from the eigendecomp were unique
            for d in D_vals:
                d_size = d.size
                uniq_d = np.unique(d)
                assert(uniq_d.size == d_size)
            return log_lik, grad[0][0]
        elif not get_grad and do_logging:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            log_lik = self.sess.run(
                    self.log_lik,
                    options=run_options,
                    run_metadata=run_metadata)

            self.profile_writer.add_run_metadata(run_metadata, "hello?")
            return log_lik, None
        else:
            return self.sess.run(self.log_lik), None

    def create_topology_log_lik(self, transition_matrix_wrappers: Dict):
        """
        Create a tensorflow graph of the likelihood calculation
        """
        self.log_lik = 0

        hazard_evt_dict, hazard_evts = self._create_hazard_dict(transition_matrix_wrappers)
        hazard_away_dict, hazard_aways = self._create_hazard_away_dict(transition_matrix_wrappers)

        singletons = CLTLikelihoodModel._get_unmasked_indels(self.topology)
        singleton_index_dict = {sg: int(i) for i, sg in enumerate(singletons)}
        singleton_log_cond_prob = self._create_log_indel_probs(singletons)

        # Store the tensorflow objects that calculate the prob of a node being in each state given the leaves
        self.L = dict()
        self.D = dict()
        self.pt_matrix = dict()
        self.trans_mats = dict()
        self.trim_probs = dict()
        for node in self.topology.traverse("postorder"):
            if node.is_leaf():
                trans_mat_w = transition_matrix_wrappers[node.node_id]
                self.L[node.node_id] = np.zeros((trans_mat_w.num_likely_states + 1, 1))
                assert len(node.state_sum.tts_list) == 1
                tts_key = trans_mat_w.key_dict[node.state_sum.tts_list[0]]
                self.L[node.node_id][tts_key] = 1
                # Convert to tensorflow usage
                self.L[node.node_id] = tf.constant(self.L[node.node_id], dtype=tf.float64)
            else:
                self.L[node.node_id] = tf.constant(1.0, dtype=tf.float64)
                for child in node.children:
                    ch_trans_mat_w = transition_matrix_wrappers[child.node_id]
                    with tf.name_scope("Transition_matrix%d" % node.node_id):
                        self.trans_mats[child.node_id] = self._create_transition_matrix(ch_trans_mat_w, hazard_evt_dict, hazard_evts, hazard_away_dict, hazard_aways)
                    # Get the trim probabilities
                    with tf.name_scope("trim_matrix%d" % node.node_id):
                        self.trim_probs[child.node_id] = self._create_trim_prob_matrix(
                                ch_trans_mat_w,
                                singleton_log_cond_prob,
                                singleton_index_dict,
                                node,
                                child)

                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = self.branch_lens[child.node_id]
                    with tf.name_scope("expm_ops%d" % node.node_id):
                        pr_matrix, _, _, D = tf_common.myexpm(self.trans_mats[child.node_id], branch_len)
                    self.D[child.node_id] = D
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

        with tf.name_scope("log_lik"):
            self.log_lik = tf.add(self.log_lik, tf.log(self.L[self.root_node_id]), name="final_log_lik")
            self.log_lik_grad = self.grad_opt.compute_gradients(
                self.log_lik,
                var_list=[self.all_vars])

    def _create_transition_matrix(self,
            matrix_wrapper: TransitionMatrixWrapper,
            hazard_evt_dict: Dict[Tuple[TargetTract], int],
            hazard_evts: Tensor,
            hazard_away_dict: Dict[Tuple[TargetTract], int],
            hazard_aways: Tensor):
        """
        Uses tensorflow to create the instantaneous transition matrix
        """
        print("NUM UNLIKELY", matrix_wrapper.num_likely_states)
        unlikely_key = matrix_wrapper.num_likely_states

        # Now fill in the matrix -- match tensorflow object with indices of instant transition matrix
        idx = 0
        index_vals = []
        for start_tts, matrix_row in matrix_wrapper.matrix_dict.items():
            start_key = matrix_wrapper.key_dict[start_tts]
            haz_away = hazard_aways[hazard_away_dict[start_tts]]

            # Hazard of staying is negative of hazard away
            index_vals.append([[start_key, start_key], -haz_away])

            # Tracks the total hazard to the likely states
            haz_to_likely = 0
            for end_tts, tt_evt in matrix_row.items():
                haz = hazard_evts[hazard_evt_dict[tt_evt]]
                end_key = matrix_wrapper.key_dict[end_tts]
                index_vals.append([[start_key, end_key], haz])
                haz_to_likely += haz
                idx += 1

            # Hazard to unlikely state is hazard away minus hazard to likely states
            index_vals.append([[start_key, unlikely_key], haz_away - haz_to_likely])

        q_matrix = tf_common.scatter_nd(
                index_vals,
                output_shape=[matrix_wrapper.num_likely_states + 1, matrix_wrapper.num_likely_states + 1],
                name="top.q_matrix")
        return q_matrix

    def _create_trim_prob_matrix(self,
            ch_trans_mat_w: TransitionMatrixWrapper,
            singleton_log_cond_prob: Tensor,
            singleton_index_dict: Dict[int, Singleton],
            node: CellLineageTree,
            child: CellLineageTree):
        """
        @param ch_trans_mat_w: the transition matrix wrapper corresponding to child node
                                (we make sure the entries in the trim prob matrix match
                                the order in ch_trans_mat)
        @param singleton_log_cond_prob: List[tensorflow array] with the log conditional prob of singletons
        @param singleton_index_dict: dictionary mapping the list index to the singleton
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
                    log_trim_probs = tf.gather(
                            params = singleton_log_cond_prob,
                            indices = [singleton_index_dict[sg] for sg in singletons])
                    log_val = tf.reduce_sum(log_trim_probs)
                    index_vals.append([[node_tts_key, child_tts_key], log_val])

        output_shape = [ch_trans_mat_w.num_likely_states + 1, ch_trans_mat_w.num_likely_states + 1]
        if index_vals:
            return tf.exp(tf_common.scatter_nd(
                index_vals,
                output_shape,
                name="top.trim_probs"))
        else:
            return tf.ones(output_shape, dtype=tf.float64)

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
        long_left_statuses = tf.constant([tt_evt.is_left_long for tt_evt in tt_evts], dtype=tf.float64)
        long_right_statuses = tf.constant([tt_evt.is_right_long for tt_evt in tt_evts], dtype=tf.float64)

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
            [[trans_mat_w.key_dict[tts], 0], ch_ordered_down_probs[ch_trans_mat_w.key_dict[tts]][0]]
            for tts in tts_list]
        down_probs = tf_common.scatter_nd(
                index_vals,
                output_shape=[trans_mat_w.num_likely_states + 1, 1],
                name="top.down_probs")
        return down_probs

    def check_grad(self, transition_matrices, epsilon=1e-10):
        orig_params = self.sess.run(self.all_vars)
        self.create_topology_log_lik(transition_matrices)
        log_lik, grad = self.get_log_lik(get_grad=True)
        print("log lik", log_lik)
        print("all grad", grad)
        for i in range(len(orig_params)):
            new_params = np.copy(orig_params)
            new_params[i] += epsilon
            self.sess.run(self.assign_all_vars, feed_dict={self.all_vars_ph: new_params})

            log_lik_eps, _ = self.get_log_lik()
            log_lik_approx = (log_lik_eps - log_lik)/epsilon
            print("index", i, " -- LOG LIK GRAD APPROX", log_lik_approx)
            print("index", i, " --                GRAD ", grad[i])

    def create_logger(self):
        self.profile_writer = tf.summary.FileWriter("_output", self.sess.graph)

    def close_logger(self):
        self.profile_writer.close()
