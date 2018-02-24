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
from cell_state import CellTypeTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract, AncState, SingletonWC, Singleton, TractRepr, DeactTract, DeactTargetsEvt, Tract
from transition_matrix import TransitionMatrixWrapper
import tf_common
from common import inv_sigmoid
from constants import PERTURB_ZERO
from bounded_poisson import BoundedPoisson

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type
    """
    NODE_ORDER = "postorder"

    def __init__(self,
            topology: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            sess: Session,
            target_lams: ndarray,
            trim_long_probs: ndarray = 0.05 * np.ones(2),
            trim_zero_prob: float = 0.5,
            trim_poissons: ndarray = 2.5 * np.ones(2),
            insert_zero_prob: float = 0.5,
            insert_poisson: float = 0.2,
            double_cut_weight: float = 0.0001,
            branch_lens: ndarray = np.array([]),
            cell_type_tree: CellTypeTree = None):
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
        self.double_cut_weight = double_cut_weight

        # Process cell type tree
        self.cell_type_tree = cell_type_tree
        cell_type_lams = []
        if cell_type_tree:
            max_cell_type = 0
            cell_type_dict = {}
            for node in cell_type_tree.traverse(self.NODE_ORDER):
                if node.is_root():
                    self.cell_type_root = node.cell_type
                cell_type_dict[node.cell_type] = node.rate
                max_cell_type = max(max_cell_type, node.cell_type)
            for i in range(max_cell_type + 1):
                cell_type_lams.append(cell_type_dict[i])
        cell_type_lams = np.array(cell_type_lams)

        # Save tensorflow session
        self.sess = sess

        # Create all the variables
        self._create_parameters(
                target_lams,
                trim_long_probs,
                [trim_zero_prob],
                trim_poissons,
                [insert_zero_prob],
                [insert_poisson],
                branch_lens,
                cell_type_lams)

        # Stores the penalty parameter
        self.pen_param_ph = tf.placeholder(tf.float64)

        self._create_hazard_node_for_simulation()

        self.grad_opt = tf.train.AdamOptimizer(learning_rate=0.01)

    def _create_parameters(self,
            target_lams: ndarray,
            trim_long_probs: ndarray,
            trim_zero_prob: float,
            trim_poissons: ndarray,
            insert_zero_prob: float,
            insert_poisson: float,
            branch_lens: ndarray,
            cell_type_lams: ndarray):
        self.all_vars = tf.Variable(
                np.concatenate([
                    # Fix the first target value -- not for optimization
                    np.log(target_lams[1:]),
                    inv_sigmoid(trim_long_probs),
                    inv_sigmoid(trim_zero_prob),
                    np.log(trim_poissons),
                    inv_sigmoid(insert_zero_prob),
                    np.log(insert_poisson),
                    np.log(branch_lens),
                    np.log(cell_type_lams)]),
                dtype=tf.float64)
        self.all_vars_ph = tf.placeholder(tf.float64, shape=self.all_vars.shape)
        self.assign_op = self.all_vars.assign(self.all_vars_ph)

        # For easy access to these model parameters
        up_to_size = target_lams.size - 1
        # First target lambda is fixed. The rest can vary. Addresses scaling issues.
        self.target_lams = tf.concat([
                    tf.constant([target_lams[0]], dtype=tf.float64),
                    tf.exp(self.all_vars[:up_to_size])],
                    axis=0)
        prev_size = up_to_size
        up_to_size += trim_long_probs.size
        self.trim_long_probs = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        self.trim_short_probs = tf.ones(1, dtype=tf.float64) - self.trim_long_probs
        prev_size = up_to_size
        up_to_size += 1
        self.trim_zero_prob = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += trim_poissons.size
        self.trim_poissons = tf.exp(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += 1
        self.insert_zero_prob = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += 1
        self.insert_poisson = tf.exp(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += branch_lens.size
        self.branch_lens = tf.exp(self.all_vars[prev_size: up_to_size])
        self.cell_type_lams = tf.exp(self.all_vars[-cell_type_lams.size:])

        # Create my poisson distributions
        self.poiss_left = tf.contrib.distributions.Poisson(self.trim_poissons[0])
        self.poiss_right = tf.contrib.distributions.Poisson(self.trim_poissons[1])
        self.poiss_insert = tf.contrib.distributions.Poisson(self.insert_poisson)

    def init_params(self,
            target_lams: ndarray,
            trim_long_probs: ndarray,
            trim_zero_prob: float,
            trim_poissons: ndarray,
            insert_zero_prob: float,
            insert_poisson: float,
            branch_lens: ndarray,
            cell_type_lams: ndarray):
        init_val = np.concatenate([
            # Fix the first target value -- not for optimization
            np.log(target_lams[1:]),
            inv_sigmoid(trim_long_probs),
            inv_sigmoid(trim_zero_prob),
            np.log(trim_poissons),
            inv_sigmoid(insert_zero_prob),
            np.log(insert_poisson),
            np.log(branch_lens),
            np.log(cell_type_lams)])
        self.sess.run(self.assign_op, feed_dict={self.all_vars_phs: init_val})

    def get_vars(self):
        return self.sess.run([
            self.target_lams,
            self.trim_long_probs,
            self.trim_zero_prob,
            self.trim_poissons,
            self.insert_zero_prob,
            self.insert_poisson,
            self.branch_lens,
            self.cell_type_lams])

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
            min_target: Tensor,
            max_target: Tensor,
            long_left_statuses: Tensor,
            long_right_statuses: Tensor):
        """
        Helpers for creating hazard in tensorflow graph and its associated gradient
        The arguments should all have the same length.
        The i-th elem in each argument corresponds to the target tract that was introduced.

        @return tensorflow tensor with the i-th value corresponding to the i-th target tract in the arguments
        """
        # Compute the hazard
        # Adding a weight for double cuts for now
        log_lambda_part = tf.log(tf.gather(self.target_lams, min_target)) + tf.log(tf.gather(self.target_lams, max_target) * self.double_cut_weight) * tf_common.not_equal_float(min_target, max_target)
        left_trim_prob = tf_common.ifelse(long_left_statuses, self.trim_long_probs[0], 1 - self.trim_long_probs[0])
        right_trim_prob = tf_common.ifelse(long_right_statuses, self.trim_long_probs[1], 1 - self.trim_long_probs[1])
        hazard = tf.exp(log_lambda_part + tf.log(left_trim_prob) + tf.log(right_trim_prob), name="hazard")
        return hazard

    def _create_hazard_deact(self,
            t0: List[int],
            t1: List[int],
            t2: List[int],
            t3: List[int]):
        """
        Helpers for creating hazard in tensorflow graph and its associated gradient
        The arguments should all have the same length.
        The i-th elem in each argument corresponds to the target tract that was introduced.

        @return tensorflow tensor with the i-th value corresponding to the i-th deactivation targets event in the arguments
        """
        #TODO: check the math here!
        # all four targets are different
        is_four_diff = tf_common.not_equal_float(t3, t2)
        left_lambda = (tf.gather(self.target_lams, t0) * self.trim_short_probs[0]
                + tf.gather(self.target_lams, t1) * self.trim_long_probs[0])
        right_lambda = (tf.gather(self.target_lams, t3) * self.trim_short_probs[1]
                + tf.gather(self.target_lams, t2) * self.trim_long_probs[1])
        four_diff_haz = right_lambda * left_lambda

        # three targets deactivated -- consecutive (2,3,4)
        left_lambda = tf.gather(self.target_lams, t0) * self.trim_short_probs[0]
        right_lambda = tf.gather(self.target_lams, t2) * self.trim_short_probs[1]

        is_three = tf_common.not_equal_float(t2, t1) * tf_common.equal_float(t3, t2)
        is_three_consec = is_three * tf_common.equal_float(t0 + 2, t2)
        lambda_three_consec = (left_lambda * tf.gather(self.target_lams, t1) * self.trim_long_probs[1]
                            + right_lambda * tf.gather(self.target_lams, t1) * self.trim_long_probs[0]
                            + self.trim_long_probs[1] * self.trim_long_probs[0] * tf.gather(self.target_lams, t1)
                            + left_lambda * right_lambda)

        # three targets deactivated -- left consecutive (2,3,5)
        is_three_left_consec = is_three * tf_common.equal_float(t0 + 1, t1)
        lambda_three_left_consec = right_lambda * (tf.gather(self.target_lams, t1) * self.trim_long_probs[0] + left_lambda)

        # three targets deactivated -- right consecutive (2,4,5)
        is_three_right_consec = is_three * tf_common.equal_float(t1 + 1, t2)
        lambda_three_right_consec = left_lambda * (tf.gather(self.target_lams, t1) * self.trim_long_probs[1] + right_lambda)

        # two targets deactivated -- consecutive
        is_two = tf_common.not_equal_float(t1, t0) * tf_common.equal_float(t2, t1)
        is_two_consec = is_two * tf_common.equal_float(t0 + 1, t1)
        lambda_two_consec = (tf.gather(self.target_lams, t0) * self.trim_short_probs[0] * self.trim_long_probs[1]
                            + tf.gather(self.target_lams, t1) * self.trim_long_probs[0] * self.trim_short_probs[1]
                            + tf.gather(self.target_lams, t0) * tf.gather(self.target_lams, t1) * self.trim_short_probs[0] * self.trim_short_probs[1])

        # two targets deactivated -- not consecutive
        is_two_not_consec = is_two * tf_common.not_equal_float(t0 + 1, t1)
        lambda_two_not_consec = tf.gather(self.target_lams, t0) * self.trim_short_probs[0] * tf.gather(self.target_lams, t1) * self.trim_short_probs[1]

        # one target deactivated
        is_one = tf_common.equal_float(t1, t0)
        lambda_one = tf.gather(self.target_lams, t0) * self.trim_short_probs[0] * self.trim_short_probs[1]

        # add them all up
        hazard = (four_diff_haz * is_four_diff
                + lambda_three_consec * is_three_consec
                + lambda_three_left_consec * is_three_left_consec
                + lambda_three_right_consec * is_three_right_consec
                + lambda_two_consec * is_two_consec
                + lambda_two_not_consec * is_two_not_consec
                + lambda_one * is_one)

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
        left_prob = check_left_max * check_left_min * tf_common.ifelse(
                tf_common.equal_float(left_trim_len, 0),
                self.poiss_left.prob(tf.constant(0, dtype=tf.float64)) + tf.constant(1, dtype=tf.float64) - self.poiss_left.cdf(max_left_trim),
                self.poiss_left.prob(left_trim_len))
        check_right_max = tf.cast(tf.less_equal(right_trim_len, max_right_trim), tf.float64)
        check_right_min = tf.cast(tf.less_equal(min_right_trim, right_trim_len), tf.float64)
        right_prob = check_right_max * check_right_min * tf_common.ifelse(
                tf_common.equal_float(right_trim_len, 0),
                self.poiss_right.prob(tf.constant(0, dtype=tf.float64)) + tf.constant(1, dtype=tf.float64) - self.poiss_right.cdf(max_right_trim),
                self.poiss_right.prob(right_trim_len))

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
        insert_len_prob = self.poiss_insert.prob(insert_lens)
        # Equal prob of all same length sequences
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

    def _create_hazard_dict(self, trans_mat_wrappers: List[TransitionMatrixWrapper]):
        """
        @param transition_matrix_wrappers: iterable with transition matrix wrappers

        @return Dict mapping the target tract introduced to its tensorflow tensor
                a tensorflow tensor with the calculations for the hazard of introducing the target tracts
                a tensorflow tensor with the calculations for the hazard of introducing the deact tracts
        """
        deact_evts = set()
        for trans_mat_wrapper in trans_mat_wrappers:
            # Get inputs ready for tensorflow
            for start_tracts, matrix_row in trans_mat_wrapper.matrix_dict.items():
                deact_evts.update(matrix_row.values())

        # Gets hazards (by tensorflow)
        target_tracts = [deact_evt for deact_evt in deact_evts if deact_evt.is_target_tract]
        deact_tracts = [deact_evt for deact_evt in deact_evts if not deact_evt.is_target_tract]

        hazard_target_tracts = self._create_hazard_target_tract_nodes(target_tracts)
        hazard_deact_tracts = self._create_hazard_deact_tract_nodes(deact_tracts)

        hazard_dict = {}
        for i, deact_evt in enumerate(target_tracts):
            hazard_dict[deact_evt] = hazard_target_tracts[i]
        for i, deact_tract in enumerate(deact_tracts):
            hazard_dict[deact_tract] = hazard_deact_tracts[i]

        return hazard_dict, hazard_target_tracts, hazard_deact_tracts

    def _create_hazard_away_dict(self, trans_mat_wrappers: List[TransitionMatrixWrapper]):
        """
        @param transition_matrix_wrappers: iterable with transition matrix wrappers

        @return Dict mapping the tuple of start target tract repr to a hazard away node
                a tensorflow tensor with the calculations for the hazard away
        """
        tract_repr_starts = set()
        for trans_mat_wrapper in trans_mat_wrappers:
            # Get inputs ready for tensorflow
            for start_tract_repr, matrix_row in trans_mat_wrapper.matrix_dict.items():
                tract_repr_starts.add(start_tract_repr)

        # Gets hazards (by tensorflow)
        tract_repr_starts = list(tract_repr_starts)
        hazard_away_nodes = self._create_hazard_away_nodes(tract_repr_starts)
        hazard_away_dict = {
                tract_repr: hazard_away_nodes[i]
                for i, tract_repr in enumerate(tract_repr_starts)}
        return hazard_away_dict, hazard_away_nodes

    def get_log_lik(self, get_grad: bool=False, do_logging: bool=False):
        """
        @return the log likelihood and the gradient, if requested
        """
        if get_grad and not do_logging:
            log_lik, grad = self.sess.run(
                    [self.log_lik, self.log_lik_grad])
            return log_lik, grad[0][0]
        elif do_logging:
            # For tensorflow logging
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # For my own checks on the d matrix and q matrix
            dkey_list = list(self.D.keys())
            D_vals = [self.D[k] for k in dkey_list]
            trans_mats_vals = [self.trans_mats[k] for k in dkey_list]
            D_cell_type_vals = [self.D_cell_type[k] for k in list(self.D_cell_type.keys())]

            if get_grad:
                log_lik, grad, Ds, q_mats, D_types = self.sess.run(
                        [self.log_lik, self.log_lik_grad, D_vals, trans_mats_vals, D_cell_type_vals],
                        options=run_options,
                        run_metadata=run_metadata)
                grad = grad[0][0]
            else:
                log_lik, Ds, q_mats, D_types = self.sess.run(
                        [self.log_lik, D_vals, trans_mats_vals, D_cell_type_vals],
                        options=run_options,
                        run_metadata=run_metadata)
                grad = None

            # Profile computation time in tensorflow
            self.profile_writer.add_run_metadata(run_metadata, "get_log_lik")

            # Quick check that all the diagonal matrix from the eigendecomp were unique
            for d, q in zip(Ds, q_mats):
                d_size = d.size
                uniq_d = np.unique(d)
                if uniq_d.size != d_size:
                    print("Uhoh. D matrix does not have unique eigenvalues. %d vs %d" % (uniq_d.size, d_size))
                    print("Q mat", np.sort(np.diag(q)))
                    print(np.sort(np.linalg.eig(q)[0]))

            return log_lik, grad
        else:
            return self.sess.run(self.log_lik), None

    def _create_cell_type_instant_matrix(self, haz_away=1e-10):
        num_leaves = tf.constant(len(self.cell_type_tree), dtype=tf.float64)
        index_vals = []
        self.num_cell_types = 0
        for node in self.cell_type_tree.traverse(self.NODE_ORDER):
            self.num_cell_types += 1
            if node.is_leaf():
                haz_node = haz_away + np.random.rand() * 1e-10
                haz_node_away = tf.constant(haz_node, dtype=tf.float64)
                index_vals.append([(node.cell_type, node.cell_type), -haz_away])
                for leaf in self.cell_type_tree:
                    if leaf.cell_type != node.cell_type:
                        index_vals.append([(node.cell_type, leaf.cell_type), haz_node_away/(num_leaves - 1)])
            else:
                tot_haz = tf.zeros([], dtype=tf.float64)
                for child in node.get_children():
                    haz_child = self.cell_type_lams[child.cell_type]
                    index_vals.append([(node.cell_type, child.cell_type), haz_child])
                    tot_haz = tf.add(tot_haz, haz_child)
                index_vals.append([(node.cell_type, node.cell_type), -tot_haz])

        q_matrix = tf_common.scatter_nd(
                index_vals,
                output_shape=[self.num_cell_types, self.num_cell_types],
                name="top.cell_type_q_matrix")
        return q_matrix

    def create_cell_type_log_lik(self):
        """
        Create a tensorflow graph of the likelihood calculation
        """
        self.cell_type_q_mat = self._create_cell_type_instant_matrix()
        # Store the tensorflow objects that calculate the prob of a node being in each state given the leaves
        self.L_cell_type = dict()
        self.D_cell_type = dict()
        # Store all the scaling terms addressing numerical underflow
        scaling_terms = []
        for node in self.topology.traverse(self.NODE_ORDER):
            if node.is_leaf():
                cell_type_one_hot = np.zeros((self.num_cell_types, 1))
                cell_type_one_hot[node.cell_state.categorical_state.cell_type] = 1
                self.L_cell_type[node.node_id] = tf.constant(cell_type_one_hot, dtype=tf.float64)
            else:
                self.L_cell_type[node.node_id] = tf.constant(1.0, dtype=tf.float64)
                for child in node.children:
                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = self.branch_lens[child.node_id]
                    with tf.name_scope("cell_type_expm_ops%d" % node.node_id):
                        pr_matrix, _, _, D = tf_common.myexpm(self.cell_type_q_mat, branch_len)
                        self.D_cell_type[child.node_id] = D
                        down_probs = tf.matmul(pr_matrix, self.L_cell_type[child.node_id])
                        self.L_cell_type[node.node_id] = tf.multiply(self.L_cell_type[node.node_id], down_probs)

                # Handle numerical underflow
                scaling_term = tf.reduce_sum(self.L_cell_type[node.node_id], name="scaling_term")
                self.L_cell_type[node.node_id] = tf.div(self.L_cell_type[node.node_id], scaling_term, name="sub_log_lik")
                scaling_terms.append(scaling_term)

        with tf.name_scope("cell_type_log_lik"):
            # Account for the scaling terms we used for handling numerical underflow
            scaling_terms = tf.stack(scaling_terms)
            self.log_lik_cell_type = tf.add(
                tf.reduce_sum(tf.log(scaling_terms), name="add_normalizer"),
                tf.log(self.L_cell_type[self.root_node_id][self.cell_type_root]),
                name="cell_type_log_lik")

    def create_topology_log_lik(self, transition_matrix_wrappers: Dict):
        """
        Create a tensorflow graph of the likelihood calculation
        """
        # Get the hazards for making the instantaneous transition matrix
        # Doing it all at once to speed up computation
        hazard_dict, hazard_target_tracts, hazard_deact_tracts = self._create_hazard_dict(
                transition_matrix_wrappers.values())
        hazard_away_dict, hazard_aways = self._create_hazard_away_dict(transition_matrix_wrappers.values())

        # Get all the conditional probabilities of the trims
        # Doing it all at once to speed up computation
        singletons = CLTLikelihoodModel._get_unmasked_indels(self.topology)
        singleton_index_dict = {sg: int(i) for i, sg in enumerate(singletons)}
        singleton_log_cond_prob = self._create_log_indel_probs(singletons)

        # Store the tensorflow objects that calculate the prob of a node being in each state given the leaves
        self.L = dict()
        self.D = dict()
        self.pt_matrix = dict()
        self.trans_mats = dict()
        self.trim_probs = dict()
        # Store all the scaling terms addressing numerical underflow
        scaling_terms = []
        for node in self.topology.traverse(self.NODE_ORDER):
            if node.is_leaf():
                trans_mat_w = transition_matrix_wrappers[node.node_id]
                self.L[node.node_id] = np.zeros((trans_mat_w.num_likely_states + 1, 1))
                assert len(node.state_sum.tract_repr_list) == 1
                tract_repr_key = trans_mat_w.key_dict[node.state_sum.tract_repr_list[0]]
                self.L[node.node_id][tract_repr_key] = 1
                # Convert to tensorflow usage
                self.L[node.node_id] = tf.constant(self.L[node.node_id], dtype=tf.float64)
            else:
                self.L[node.node_id] = tf.constant(1.0, dtype=tf.float64)
                for child in node.children:
                    ch_trans_mat_w = transition_matrix_wrappers[child.node_id]
                    with tf.name_scope("Transition_matrix%d" % node.node_id):
                        self.trans_mats[child.node_id] = self._create_transition_matrix(
                                ch_trans_mat_w,
                                hazard_dict,
                                hazard_away_dict)
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
                    with tf.name_scope("recurse%d" % node.node_id):
                        ch_ordered_down_probs = tf.matmul(
                                tf.multiply(self.pt_matrix[child.node_id], self.trim_probs[child.node_id]),
                                self.L[child.node_id])

                    with tf.name_scope("rearrange%d" % node.node_id):
                        if not node.is_root():
                            # Reorder summands according to node's numbering of tract_repr states
                            trans_mat_w = transition_matrix_wrappers[node.node_id]

                            down_probs = CLTLikelihoodModel._reorder_likelihoods(
                                    ch_ordered_down_probs,
                                    node.state_sum.tract_repr_list,
                                    trans_mat_w,
                                    ch_trans_mat_w)

                            self.L[node.node_id] = tf.multiply(self.L[node.node_id], down_probs)
                        else:
                            # For the root node, we just want the probability where the root node is unmodified
                            # No need to reorder
                            ch_id = ch_trans_mat_w.key_dict[()]
                            self.L[node.node_id] = tf.multiply(self.L[node.node_id], ch_ordered_down_probs[ch_id])

                # Handle numerical underflow
                scaling_term = tf.reduce_sum(self.L[node.node_id], name="scaling_term")
                self.L[node.node_id] = tf.div(self.L[node.node_id], scaling_term, name="sub_log_lik")
                scaling_terms.append(scaling_term)

        with tf.name_scope("alleles_log_lik"):
            # Account for the scaling terms we used for handling numerical underflow
            scaling_terms = tf.stack(scaling_terms)
            self.log_lik_alleles = tf.add(
                tf.reduce_sum(tf.log(scaling_terms), name="add_normalizer"),
                tf.log(self.L[self.root_node_id]),
                name="alleles_log_lik")

    def create_log_lik(self, transition_matrix_wrappers: Dict):
        self.log_lik_cell_type = tf.zeros([])
        self.create_topology_log_lik(transition_matrix_wrappers)
        if self.cell_type_tree is None:
            self.log_lik = self.log_lik_alleles
        else:
            self.create_cell_type_log_lik()
            self.log_lik = self.log_lik_cell_type + self.log_lik_alleles

        self.log_lik_grad = self.grad_opt.compute_gradients(
            self.log_lik,
            var_list=[self.all_vars])

        self.pen_log_lik = tf.add(
                self.log_lik,
                -self.pen_param_ph * tf.reduce_sum(tf.pow(self.branch_lens, 2)),
                name="final_pen_log_lik")
        self.pen_log_lik_grad = self.grad_opt.compute_gradients(
            self.pen_log_lik,
            var_list=[self.all_vars])

        self.train_op = self.grad_opt.minimize(-self.pen_log_lik, var_list=self.all_vars)

    def _create_transition_matrix(self,
            matrix_wrapper: TransitionMatrixWrapper,
            hazard_dict: Dict[Tract, Tensor],
            hazard_away_dict: Dict[TractRepr, int]):
        """
        Uses tensorflow to create the instantaneous transition matrix
        """
        #print("NUM UNLIKELY", matrix_wrapper.num_likely_states)
        unlikely_key = matrix_wrapper.num_likely_states

        # Now fill in the matrix -- match tensorflow object with indices of instant transition matrix
        idx = 0
        index_vals = []
        for start_tract_repr, matrix_row in matrix_wrapper.matrix_dict.items():
            start_key = matrix_wrapper.key_dict[start_tract_repr]
            haz_away = hazard_away_dict[start_tract_repr]

            # Hazard of staying is negative of hazard away
            index_vals.append([[start_key, start_key], -haz_away])

            if start_tract_repr == (TargetTract(0, 0, self.num_targets - 1, self.num_targets - 1),):
                # This is an annoying case where we have two zero eigenvalues...
                index_vals.append([[unlikely_key, unlikely_key], -PERTURB_ZERO])
                index_vals.append([[unlikely_key, start_key], PERTURB_ZERO])
                continue

            # Tracks the total hazard to the likely states
            haz_to_likely = 0
            for end_tract_repr, tt_evt in matrix_row.items():
                haz = hazard_dict[tt_evt]
                end_key = matrix_wrapper.key_dict[end_tract_repr]
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

        for node_tract_repr in node.state_sum.tract_repr_list:
            node_tract_repr_key = ch_trans_mat_w.key_dict[node_tract_repr]
            for child_tract_repr in child.state_sum.tract_repr_list:
                child_tract_repr_key = ch_trans_mat_w.key_dict[child_tract_repr]
                diff_target_tracts = node_tract_repr.diff(child_tract_repr)
                singletons = CLTLikelihoodModel.get_matching_singletons(child.anc_state, diff_target_tracts)

                if singletons:
                    log_trim_probs = tf.gather(
                            params = singleton_log_cond_prob,
                            indices = [singleton_index_dict[sg] for sg in singletons])
                    log_val = tf.reduce_sum(log_trim_probs)
                    index_vals.append([[node_tract_repr_key, child_tract_repr_key], log_val])

        output_shape = [ch_trans_mat_w.num_likely_states + 1, ch_trans_mat_w.num_likely_states + 1]
        if index_vals:
            return tf.exp(tf_common.scatter_nd(
                index_vals,
                output_shape,
                name="top.trim_probs"))
        else:
            return tf.ones(output_shape, dtype=tf.float64)

    def get_hazards(self, tt_evts: List[TargetTract]):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        hazards = self.sess.run(
                self.hazard,
                feed_dict={
                    self.targets_ph: [
                        [tt_evt.min_target, tt_evt.max_target] for tt_evt in tt_evts],
                    self.long_status_ph: [
                        [tt_evt.is_left_long, tt_evt.is_right_long] for tt_evt in tt_evts]})
        return hazards

    def get_hazard(self, tt_evt: TargetTract):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        return self.get_hazards([tt_evt])[0]

    def _get_hazard_masks(self, tract_repr: TractRepr):
        """
        @param tract_repr: the target tract repr that we would like to process

        @return two lists, one for left trims and one for right trims.
                each list is composed of values [0,1,2] matching each target.
                0 = no trim at that target (in that direction) can be performed
                1 = only short trim at that target (in that dir) is allowed
                2 = short and long trims are allowed at that target

                We assume no long left trims are allowed at target 0 and any target T
                where T - 1 is deactivated. No long right trims are allowed at the last
                target and any target T where T + 1 is deactivated.
        """
        if len(tract_repr) == 0:
            # This is the original barcode
            left_hazard_list = [1] + [2] * (self.num_targets - 1)
            right_hazard_list = [2] * (self.num_targets - 1) + [1]
        else:
            # This is an allele with some indel somewhere

            # Process the section before the first indel
            if tract_repr[0].min_deact_target > 1:
                left_hazard_list = [1] + [2] * (tract_repr[0].min_deact_target - 1)
                right_hazard_list = [2] * (tract_repr[0].min_deact_target - 1) + [1]
            elif tract_repr[0].min_deact_target == 1:
                left_hazard_list = [1]
                right_hazard_list = [1]
            else:
                left_hazard_list = []
                right_hazard_list = []

            left_hazard_list += [0] * (tract_repr[0].max_deact_target - tract_repr[0].min_deact_target + 1)
            right_hazard_list += [0] * (tract_repr[0].max_deact_target - tract_repr[0].min_deact_target + 1)

            # Process the sections with the indels
            for i, tract in enumerate(tract_repr[1:]):
                if tract_repr[i].max_deact_target + 1 < tract.min_deact_target - 1:
                    left_hazard_list += [1] + [2] * (tract.min_deact_target - tract_repr[i].max_deact_target - 2)
                    right_hazard_list += [2] * (tract.min_deact_target - tract_repr[i].max_deact_target - 2) + [1]
                elif tract_repr[i].max_deact_target + 1 == tract.min_deact_target - 1:
                    # Single target, short left and right
                    left_hazard_list += [1]
                    right_hazard_list += [1]
                left_hazard_list += [0] * (tract.max_deact_target - tract.min_deact_target + 1)
                right_hazard_list += [0] * (tract.max_deact_target - tract.min_deact_target + 1)

            # Process the section after all the indels
            if tract_repr[-1].max_deact_target < self.num_targets - 2:
                left_hazard_list += [1] + [2] * (self.num_targets - tract_repr[-1].max_deact_target - 2)
                right_hazard_list += [2] * (self.num_targets - tract_repr[-1].max_deact_target - 2) + [1]
            elif tract_repr[-1].max_deact_target == self.num_targets - 2:
                left_hazard_list += [1]
                right_hazard_list += [1]

        assert(len(left_hazard_list) == self.num_targets)
        assert(len(right_hazard_list) == self.num_targets)
        return left_hazard_list, right_hazard_list

    def _create_hazard_target_tract_nodes(self, tt_evts: List[TargetTract]):
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

    def _create_hazard_deact_tract_nodes(self, deact_targs_evts: List[DeactTargetsEvt]):
        """
        @return tensorfow array of the hazard of introducing each target tract in `deact_targs_evts`
        """
        t0 = tf.constant([deact_targs_evt[0] for deact_targs_evt in deact_targs_evts], dtype=tf.int32)
        t1 = tf.constant([
            deact_targs_evt[1] if len(deact_targs_evt) >= 2 else deact_targs_evt[-1]
            for deact_targs_evt in deact_targs_evts],
            dtype=tf.int32)
        t2 = tf.constant([
            deact_targs_evt[2] if len(deact_targs_evt) >= 3 else deact_targs_evt[-1]
            for deact_targs_evt in deact_targs_evts],
            dtype=tf.int32)
        t3 = tf.constant([
            deact_targs_evt[3] if len(deact_targs_evt) == 4 else deact_targs_evt[-1]
            for deact_targs_evt in deact_targs_evts],
            dtype=tf.int32)

        # Compute the hazard
        hazard_nodes = self._create_hazard_deact(t0, t1, t2, t3)
        return hazard_nodes

    def get_hazard_away(self, tract_repr: TractRepr):
        haz_away = self._create_hazard_away_nodes([tract_repr])
        return self.sess.run(haz_away)[0]

    def _create_hazard_away_nodes(self, tract_repr_list: List[TractRepr]):
        """
        @return tensorfow array of the hazard away from each target tract repr in `tract_repr_list`
        """
        left_trimmables = []
        right_trimmables = []
        for tract_repr in tract_repr_list:
            left_m, right_m = self._get_hazard_masks(tract_repr)
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
    def get_matching_singletons(anc_state: AncState, tract_repr: TractRepr):
        """
        @return the list of singletons in `anc_state` that match any target tract in `tract_repr`
        """
        # Get only the singletons
        sg_only_anc_state = [
            indel_set for indel_set in anc_state.indel_set_list if indel_set.__class__ == SingletonWC]

        # Now figure out which ones match
        sg_idx = 0
        tract_repr_idx = 0
        n_sg = len(sg_only_anc_state)
        n_tract_repr = len(tract_repr)
        matching_sgs = []
        while sg_idx < n_sg and tract_repr_idx < n_tract_repr:
            cur_tract = tract_repr[tract_repr_idx]
            sgwc = sg_only_anc_state[sg_idx]
            sg_tt = TargetTract(sgwc.min_deact_target, sgwc.min_target, sgwc.max_target, sgwc.max_deact_target)

            if cur_tract.max_deact_target < sg_tt.min_deact_target:
                tract_repr_idx += 1
                continue
            elif sg_tt.max_deact_target < cur_tract.min_deact_target:
                sg_idx += 1
                continue

            # Overlapping now
            if sg_tt == cur_tract:
               matching_sgs.append(sgwc.get_singleton())
            sg_idx += 1
            tract_repr_idx += 1

        return matching_sgs

    @staticmethod
    def get_possible_deact_targets_evts(active_any_targs: List[int]):
        """
        @param active_any_targs: a list of active targets that can be cut with any trim
        @return a set of possible target tracts
        """
        # create all possible deact tracts by combining possible start and ends
        deact_evts = set()
        # TODO: make this more efficient. it's just borrowoing all the stuff from the other
        #       function right now
        target_tract_evts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
        for tt in target_tract_evts:
            deactivated_targs = sorted(set(tt))
            deact_evts.add(DeactTargetsEvt(*deactivated_targs))

        return deact_evts

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
                    for node_tract_repr in node.state_sum.tract_repr_list:
                        for child_tract_repr in child.state_sum.tract_repr_list:
                            diff_target_tracts = node_tract_repr.diff(child_tract_repr)
                            matching_sgs = CLTLikelihoodModel.get_matching_singletons(
                                    child.anc_state,
                                    diff_target_tracts)
                            singletons.update(matching_sgs)
        return singletons

    @staticmethod
    def _reorder_likelihoods(
            ch_ordered_down_probs,
            tract_repr_list:List[TractRepr],
            trans_mat_w: TransitionMatrixWrapper,
            ch_trans_mat_w: TransitionMatrixWrapper):
        """
        @param ch_ordered_down_probs: the Tensorflow array to be re-ordered
        @param tract_repr_list: list of target tract reprs to include in the vector
                        rest can be set to zero
        @param node_trans_mat: provides the desired ordering
        @param ch_trans_mat: provides the ordering used in vec_lik

        @return the reordered version of vec_lik according to the order in node_trans_mat
        """
        index_vals = [[
                [trans_mat_w.key_dict[tract_repr], 0],
                ch_ordered_down_probs[ch_trans_mat_w.key_dict[tract_repr]][0]]
            for tract_repr in tract_repr_list]
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
