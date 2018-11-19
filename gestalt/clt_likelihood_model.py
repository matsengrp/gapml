import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import tensorflow_probability as tfp

from typing import List, Dict
from numpy import ndarray
from tensorflow import Session

import collapsed_tree
from cell_state import CellTypeTree
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import Singleton
from target_status import TargetStatus
from transition_wrapper_maker import TransitionWrapper
import tf_common
from common import inv_sigmoid, assign_rand_tree_lengths
from constants import PERTURB_ZERO
from optim_settings import KnownModelParams
from clt_chronos_estimator import CLTChronosEstimator

from profile_support import profile


class CLTLikelihoodModel:
    """
    Stores model parameters and branch lengths
    """
    NODE_ORDER = "preorder"

    def __init__(
            self,
            topology: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            sess: Session,
            known_params: KnownModelParams,
            target_lams: ndarray,
            target_lam_decay_rate: ndarray = np.array([1e-10]),
            boost_softmax_weights: ndarray = np.ones(3),
            trim_long_factor: ndarray = 0.05 * np.ones(2),
            trim_zero_probs: ndarray = 0.5 * np.ones(2),
            trim_short_nbinom_m: ndarray = 4 * np.ones(2),
            trim_short_nbinom_logits: ndarray = np.zeros(2),
            trim_long_nbinom_m: ndarray = 2.5 * np.ones(2),
            trim_long_nbinom_logits: ndarray = np.zeros(2),
            insert_zero_prob: ndarray = np.array([0.5]),
            insert_nbinom_m: ndarray = np.array([1]),
            insert_nbinom_logit: ndarray = np.ones(1),
            double_cut_weight: ndarray = np.array([1.0]),
            branch_len_inners: ndarray = np.array([]),
            branch_len_offsets_proportion: ndarray = np.array([]),
            scratch_dir: str = "_output/scratch",
            cell_type_tree: CellTypeTree = None,
            tot_time: float = 1,
            tot_time_extra: float = 0.1,
            boost_len: int = 1,
            abundance_weight: float = 0,
            step_size: float = 0.01):
        """
        @param topology: provides a topology only (ignore any branch lengths in this tree)
        @param boost_softmax_weights: vals to plug into softmax to get the probability of boosting
            insertion, left del, and right del
        @param double_cut_weight: a weight for inter-target indels
        @param target_lams: target lambda rates
        @param trim_long_factor: the scaling factor for the trim long hazard rate. assumed to be less than 1
        @param tot_time: total height of the tree
        @param step_size: the step size to initialize for the adam optimizer
        """
        assert known_params.target_lams or known_params.tot_time

        self.topology = topology
        self.num_nodes = 0
        if self.topology:
            self.root_node_id = self.topology.node_id
            assert self.root_node_id == 0
            self.num_nodes = self.topology.get_num_nodes()
        self.bcode_meta = bcode_meta
        self.cell_type_tree = cell_type_tree
        self.targ_stat_transitions_dict, _ = TargetStatus.get_all_transitions(self.bcode_meta)

        self.num_targets = bcode_meta.n_targets
        self.boost_len = boost_len
        self.abundance_weight = abundance_weight
        assert abundance_weight >= 0 and abundance_weight <= 1
        assert boost_len == 1

        # Save tensorflow session
        self.sess = sess

        self.known_params = known_params
        self.scratch_dir = scratch_dir

        # Stores the penalty parameters
        self.branch_pen_param_ph = tf.placeholder(tf.float64)
        self.crazy_pen_param_ph = tf.placeholder(tf.float64)
        self.target_lam_pen_param_ph = tf.placeholder(tf.float64)

        if branch_len_inners.size == 0:
            branch_len_inners = np.random.rand(self.num_nodes) * 0.3
        if branch_len_offsets_proportion.size == 0:
            branch_len_offsets_proportion = np.random.rand(self.num_nodes) * 0.5

        assert np.all(trim_long_factor < 1)
        # Create all the variables
        self._create_known_parameters(
                target_lams,
                target_lam_decay_rate,
                double_cut_weight,
                boost_softmax_weights,
                trim_long_factor,
                trim_zero_probs,
                trim_short_nbinom_m,
                trim_short_nbinom_logits,
                trim_long_nbinom_m,
                trim_long_nbinom_logits,
                insert_zero_prob,
                insert_nbinom_m,
                insert_nbinom_logit,
                branch_len_inners,
                branch_len_offsets_proportion,
                tot_time_extra)
        self._create_unknown_parameters(
                target_lams,
                target_lam_decay_rate,
                double_cut_weight,
                boost_softmax_weights,
                trim_long_factor,
                trim_zero_probs,
                trim_short_nbinom_m,
                trim_short_nbinom_logits,
                trim_long_nbinom_m,
                trim_long_nbinom_logits,
                insert_zero_prob,
                insert_nbinom_m,
                insert_nbinom_logit,
                branch_len_inners,
                branch_len_offsets_proportion,
                tot_time_extra)

        self.tot_time = tf.constant(tot_time, dtype=tf.float64)

        # Calculcate the hazards for all the target tracts beforehand. Speeds up computation in the future.
        self.target_tract_hazards, self.target_tract_dict = self._create_all_target_tract_hazards()
        # Dictionary for storing hazards between target statuses -- assuming all moves are possible
        self.targ_stat_transition_hazards_dict = {
            start_target_status: {} for start_target_status in self.targ_stat_transitions_dict.keys()}
        # Calculate hazard for transitioning away from all target statuses beforehand. Speeds up future computation.
        self.hazard_away_dict = self._create_hazard_away_dict()

        if self.topology:
            assert not self.topology.is_leaf()

            # Process branch len offsets and inners
            if self.known_params.branch_lens:
                self.branch_len_inners = tf.scatter_nd(
                    self.known_params.branch_len_inners_idxs,
                    self.branch_len_inners_known,
                    (self.num_nodes,)) + tf.scatter_nd(
                    self.known_params.branch_len_inners_unknown_idxs,
                    self.branch_len_inners_unknown,
                    (self.num_nodes,))
                self.branch_len_offsets_proportion = tf.scatter_nd(
                    self.known_params.branch_len_offsets_proportion_idxs,
                    self.branch_len_offsets_proportion_known,
                    (self.num_nodes,)) + tf.scatter_nd(
                    self.known_params.branch_len_offsets_proportion_unknown_idxs,
                    self.branch_len_offsets_proportion_unknown,
                    (self.num_nodes,))
            else:
                self.branch_len_inners = self.branch_len_inners_unknown
                self.branch_len_offsets_proportion = self.branch_len_offsets_proportion_unknown

            self._create_distance_to_root_dict()
            self.branch_lens = self._create_branch_lens()
        else:
            self.branch_len_inners = []
            self.branch_len_offsets_proportion = []

        self.step_size = step_size
        self.adam_opt = tf.train.AdamOptimizer(learning_rate=self.step_size)

    def _create_known_parameters(
            self,
            target_lams: ndarray,
            target_lam_decay_rate: ndarray,
            double_cut_weight: ndarray,
            boost_softmax_weights: ndarray,
            trim_long_factor: ndarray,
            trim_zero_probs: ndarray,
            trim_short_nbinom_m: ndarray,
            trim_short_nbinom_logits: ndarray,
            trim_long_nbinom_m: ndarray,
            trim_long_nbinom_logits: ndarray,
            insert_zero_prob: ndarray,
            insert_nbinom_m: ndarray,
            insert_nbinom_logit: ndarray,
            branch_len_inners: ndarray,
            branch_len_offsets_proportion: ndarray,
            tot_time_extra: float):
        """
        Creates the tensorflow nodes for each of the model parameters
        """
        known_model_params = np.concatenate([
                    [0] if self.known_params.tot_time else [],
                    target_lams if self.known_params.target_lams else [],
                    target_lam_decay_rate if self.known_params.target_lam_decay_rate else [],
                    double_cut_weight if self.known_params.double_cut_weight else [],
                    trim_long_factor if self.known_params.trim_long_factor else [],
                    trim_short_nbinom_m if self.known_params.indel_dists else [],
                    trim_short_nbinom_logits if self.known_params.indel_dists else [],
                    trim_long_nbinom_m if self.known_params.indel_dists else [],
                    trim_long_nbinom_logits if self.known_params.indel_dists else [],
                    insert_nbinom_m if self.known_params.indel_dists else [],
                    insert_nbinom_logit if self.known_params.indel_dists else [],
                    boost_softmax_weights if self.known_params.indel_params else [],
                    trim_zero_probs if self.known_params.indel_params else [],
                    insert_zero_prob if self.known_params.indel_params else [],
                    branch_len_inners[self.known_params.branch_len_inners] if self.known_params.branch_lens else [],
                    branch_len_offsets_proportion[self.known_params.branch_len_offsets_proportion] if self.known_params.branch_lens else []])
        self.known_vars = tf.Variable(known_model_params, dtype=tf.float64)
        self.known_vars_ph = tf.placeholder(tf.float64, shape=self.known_vars.shape)
        self.assign_known_op = self.known_vars.assign(self.known_vars_ph)

        # For easy access to these model parameters
        up_to_size = 0
        if self.known_params.tot_time:
            up_to_size = 1
            self.tot_time_extra = self.known_vars[0]
        prev_size = up_to_size
        if self.known_params.target_lams:
            up_to_size += target_lams.size
            self.target_lams = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.target_lam_decay_rate:
            up_to_size += 1
            self.target_lam_decay_rate = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.double_cut_weight:
            up_to_size += 1
            self.double_cut_weight = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.trim_long_factor:
            up_to_size += trim_long_factor.size
            self.trim_long_factor = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.indel_dists:
            up_to_size += trim_short_nbinom_m.size
            self.trim_short_nbinom_m = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += trim_short_nbinom_logits.size
            self.trim_short_nbinom_logits = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += trim_long_nbinom_m.size
            self.trim_long_nbinom_m = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += trim_long_nbinom_logits.size
            self.trim_long_nbinom_logits = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += 1
            self.insert_nbinom_m = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += 1
            self.insert_nbinom_logit = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.indel_params:
            up_to_size += boost_softmax_weights.size
            self.boost_softmax_weights = self.known_vars[prev_size: up_to_size]
            self.boost_probs = tf.nn.softmax(self.boost_softmax_weights)
            prev_size = up_to_size
            up_to_size += trim_zero_probs.size
            self.trim_zero_probs = self.known_vars[prev_size: up_to_size]
            self.trim_zero_prob_left = self.trim_zero_probs[0]
            self.trim_zero_prob_right = self.trim_zero_probs[1]
            prev_size = up_to_size
            up_to_size += 1
            self.insert_zero_prob = self.known_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if self.known_params.branch_lens:
            up_to_size += np.sum(self.known_params.branch_len_inners)
            self.branch_len_inners_known = self.known_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += np.sum(self.known_params.branch_len_offsets_proportion)
            self.branch_len_offsets_proportion_known = self.known_vars[prev_size: up_to_size]

    def _create_trim_insert_distributions(self, num_singletons: int):
        """
        Creates the basic trim + insert helper distributions

        NOTE: Requires the number of singletons because tensorflow refuses to properly broadcast for
        the negative binomial distribution for some awful reason
        """
        def make_del_dist(nbinom_m, nbinom_logits):
            return [
                tfp.distributions.NegativeBinomial(
                    [nbinom_m[0]] * num_singletons,
                    logits=[nbinom_logits[0]] * num_singletons),
                tfp.distributions.NegativeBinomial(
                    [nbinom_m[1]] * num_singletons,
                    logits=[nbinom_logits[1]] * num_singletons)]
        self.del_short_dist = make_del_dist(self.trim_short_nbinom_m, self.trim_short_nbinom_logits)
        self.del_long_dist = make_del_dist(self.trim_long_nbinom_m, self.trim_long_nbinom_logits)
        self.insert_dist = tfp.distributions.NegativeBinomial(self.insert_nbinom_m, logits=self.insert_nbinom_logit)

    def _create_unknown_parameters(
            self,
            target_lams: ndarray,
            target_lam_decay_rate: ndarray,
            double_cut_weight: ndarray,
            boost_softmax_weights: ndarray,
            trim_long_factor: ndarray,
            trim_zero_probs: ndarray,
            trim_short_nbinom_m: ndarray,
            trim_short_nbinom_logits: ndarray,
            trim_long_nbinom_m: ndarray,
            trim_long_nbinom_logits: ndarray,
            insert_zero_prob: ndarray,
            insert_nbinom_m: ndarray,
            insert_nbinom_logit: ndarray,
            branch_len_inners: ndarray,
            branch_len_offsets_proportion: ndarray,
            tot_time_extra: float):
        """
        Creates the tensorflow nodes for each of the model parameters
        """
        assert boost_softmax_weights.size == 3

        # Fix the first target value -- not for optimization
        model_params = np.concatenate([
                    [] if self.known_params.tot_time else np.log([tot_time_extra]),
                    [] if self.known_params.target_lams else np.log(target_lams),
                    [] if self.known_params.target_lam_decay_rate else inv_sigmoid(target_lam_decay_rate),
                    [] if self.known_params.double_cut_weight else np.log(double_cut_weight),
                    [] if self.known_params.trim_long_factor else inv_sigmoid(trim_long_factor),
                    [] if self.known_params.indel_dists else np.log(trim_short_nbinom_m),
                    [] if self.known_params.indel_dists else trim_short_nbinom_logits,
                    [] if self.known_params.indel_dists else np.log(trim_long_nbinom_m),
                    [] if self.known_params.indel_dists else trim_long_nbinom_logits,
                    [] if self.known_params.indel_dists else np.log(insert_nbinom_m),
                    [] if self.known_params.indel_dists else insert_nbinom_logit,
                    [] if self.known_params.indel_params else boost_softmax_weights,
                    [] if self.known_params.indel_params else inv_sigmoid(trim_zero_probs),
                    [] if self.known_params.indel_params else inv_sigmoid(insert_zero_prob),
                    np.log(branch_len_inners[self.known_params.branch_len_inners_unknown] if self.known_params.branch_lens else branch_len_inners),
                    inv_sigmoid(branch_len_offsets_proportion[self.known_params.branch_len_offsets_proportion_unknown] if self.known_params.branch_lens else branch_len_offsets_proportion)])
        self.all_vars = tf.Variable(model_params, dtype=tf.float64)
        self.all_vars_ph = tf.placeholder(tf.float64, shape=self.all_vars.shape)
        self.assign_op = self.all_vars.assign(self.all_vars_ph)

        # For easy access to these model parameters
        up_to_size = 0
        if not self.known_params.tot_time:
            up_to_size = 1
            self.tot_time_extra = tf.exp(self.all_vars[0])
        prev_size = up_to_size
        if not self.known_params.target_lams:
            up_to_size += target_lams.size
            self.target_lams = tf.exp(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        if not self.known_params.target_lam_decay_rate:
            up_to_size += 1
            self.target_lam_decay_rate = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        if not self.known_params.double_cut_weight:
            up_to_size += 1
            self.double_cut_weight = tf.exp(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        if not self.known_params.trim_long_factor:
            up_to_size += trim_long_factor.size
            self.trim_long_factor = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        if not self.known_params.indel_dists:
            up_to_size += trim_short_nbinom_m.size
            self.trim_short_nbinom_m = tf.exp(self.all_vars[prev_size: up_to_size])
            prev_size = up_to_size
            up_to_size += trim_short_nbinom_logits.size
            self.trim_short_nbinom_logits = self.all_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += trim_long_nbinom_m.size
            self.trim_long_nbinom_m = tf.exp(self.all_vars[prev_size: up_to_size])
            prev_size = up_to_size
            up_to_size += trim_long_nbinom_logits.size
            self.trim_long_nbinom_logits = self.all_vars[prev_size: up_to_size]
            prev_size = up_to_size
            up_to_size += 1
            self.insert_nbinom_m = tf.exp(self.all_vars[prev_size: up_to_size])
            prev_size = up_to_size
            up_to_size += 1
            self.insert_nbinom_logit = self.all_vars[prev_size: up_to_size]
        prev_size = up_to_size
        if not self.known_params.indel_params:
            up_to_size += boost_softmax_weights.size
            self.boost_softmax_weights = self.all_vars[prev_size: up_to_size]
            self.boost_probs = tf.nn.softmax(self.boost_softmax_weights)
            prev_size = up_to_size
            up_to_size += trim_zero_probs.size
            self.trim_zero_probs = tf.sigmoid(self.all_vars[prev_size: up_to_size])
            self.trim_zero_prob_left = self.trim_zero_probs[0]
            self.trim_zero_prob_right = self.trim_zero_probs[1]
            prev_size = up_to_size
            up_to_size += 1
            self.insert_zero_prob = tf.sigmoid(self.all_vars[prev_size: up_to_size])
        self.all_but_branch_lens = self.all_vars[:up_to_size]
        prev_size = up_to_size
        up_to_size += np.sum(self.known_params.branch_len_inners_unknown) if self.known_params.branch_lens else branch_len_inners.size
        self.branch_len_inners_unknown = tf.exp(self.all_vars[prev_size: up_to_size])
        prev_size = up_to_size
        up_to_size += np.sum(self.known_params.branch_len_offsets_proportion_unknown) if self.known_params.branch_lens else branch_len_offsets_proportion.size
        self.branch_len_offsets_proportion_unknown = tf.sigmoid(self.all_vars[prev_size: up_to_size])

    def _create_distance_to_root_dict(self):
        """
        Create distance to root tensors for all internal nodes
        as well as branch length offsets
        """
        # Preprocess the special case where the branch length inner is know but the proportion is not for a child node of a bifurcating node.
        # This is a weird flag because for bifurcating nodes, the proportion typically doesn't mean anything; so variables should be both known or both unknown.
        # In this special weird case, it means that the proportion indicates the location of the parent node.
        # This parent node is "implicitly" defined by the proportion -- its own inner and proportion variables are not relevant and are completely ignored.
        # The reason we have these implicit nodes is to deal with branch length constraints -- we know the node's offset from that grandparent but don't know
        # where to place the parent node in between.
        self.dist_to_root = {self.root_node_id: tf.constant(0, dtype=tf.float64)}
        branch_offsets = {self.root_node_id: tf.constant(0, dtype=tf.float64)}
        # Using preorder is important so we don't override dist_to_root incorrectly
        for node in self.topology.get_descendants("preorder"):
            if hasattr(node, 'implicit_child') and node.implicit_child is not None:
                # This node's specifications are implicitly defined by its child -- its branch length is proportion[node] * branch_length[`node.implicit_child`]
                # `node.implicit_child` should be one of the children of `node`
                assert len(node.get_children()) <= 2

                # First we get the distance to root for all the nodes on the path up to the root.
                nodes_on_path_to_root = [node.up]
                while nodes_on_path_to_root[0].up is not None:
                    nodes_on_path_to_root = [nodes_on_path_to_root[0].up] + nodes_on_path_to_root
                for path_node in nodes_on_path_to_root:
                    if path_node.node_id not in self.dist_to_root:
                        self.dist_to_root[path_node.node_id] = self.dist_to_root[path_node.up.node_id] + self.branch_len_inners[path_node.node_id]

                # dist_to_root of `node.implicit_child` is defined by itself and the parent of the `node`
                # we ignore the implicit node
                if node.implicit_child.is_leaf():
                    self.dist_to_root[node.implicit_child.node_id] = self.tot_time
                else:
                    self.dist_to_root[node.implicit_child.node_id] = self.dist_to_root[node.up.node_id] + self.branch_len_inners[node.implicit_child.node_id]

                # The proportion in the implicit node is used to decide the branch length of the implicit node
                # To get the branch length of the implicit node, multiply its proportion with the branch length of the `implicit_child`.
                br_len_inner_child_orig = self.dist_to_root[node.implicit_child.node_id] - self.dist_to_root[node.up.node_id]
                if not node.up.is_resolved_multifurcation():
                    br_offset_child_orig = self.branch_len_offsets_proportion[node.implicit_child.node_id]
                    branch_offsets[node.node_id] = br_len_inner_child_orig * br_offset_child_orig
                    br_len_child_orig = br_len_inner_child_orig * (1 - br_offset_child_orig)
                    br_len = br_len_child_orig * self.branch_len_offsets_proportion[node.node_id]
                else:
                    branch_offsets[node.node_id] = tf.constant(0, dtype=tf.float64)
                    br_len = br_len_inner_child_orig * self.branch_len_offsets_proportion[node.node_id]
                branch_offsets[node.implicit_child.node_id] = tf.constant(0, dtype=tf.float64)

                self.dist_to_root[node.node_id] = self.dist_to_root[node.up.node_id] + branch_offsets[node.node_id] + br_len

        # Now that we've handled the special implicit nodes, we are ready to calculate the rest of the dist_to_root distances
        for node in self.topology.get_descendants("preorder"):
            if node.node_id in self.dist_to_root:
                continue

            if node.is_leaf():
                self.dist_to_root[node.node_id] = self.tot_time
            else:
                self.dist_to_root[node.node_id] = self.dist_to_root[node.up.node_id] + self.branch_len_inners[node.node_id]

        # This also calculates the offset on the spines for nodes that are children of multifurcating nodes
        for node in self.topology.get_descendants("preorder"):
            if node.node_id in branch_offsets:
                continue

            if not node.up.is_resolved_multifurcation():
                if node.is_leaf():
                    # A leaf node, use its offset to determine branch length
                    br_len_inner = self.tot_time - self.dist_to_root[node.up.node_id]
                    branch_offsets[node.node_id] = self.branch_len_offsets_proportion[node.node_id] * br_len_inner
                else:
                    branch_offsets[node.node_id] = self.branch_len_inners[node.node_id] * self.branch_len_offsets_proportion[node.node_id]
            else:
                branch_offsets[node.node_id] = tf.constant(0, dtype=tf.float64)
        self.branch_len_offsets = tf.stack([branch_offsets[j] for j in range(self.num_nodes)])

    def _create_branch_lens(self):
        """
        Create the branch length variable
        """
        branch_lens_dict = []
        for node in self.topology.traverse("preorder"):
            if node.is_root():
                continue

            if not node.up.is_resolved_multifurcation():
                if node.is_leaf():
                    # A leaf node, use its offset to determine branch length
                    br_len = self.tot_time - self.dist_to_root[node.up.node_id] - self.branch_len_offsets[node.node_id]
                else:
                    # Internal node -- use the branch length minus offset to specify the branch length (in the bifurcating tree)
                    # But use the offset + branch length to determine distance from root
                    br_len = self.dist_to_root[node.node_id] - self.dist_to_root[node.up.node_id] - self.branch_len_offsets[node.node_id]
            else:
                if node.is_leaf():
                    br_len = self.tot_time - self.dist_to_root[node.up.node_id]
                else:
                    br_len = self.dist_to_root[node.node_id] - self.dist_to_root[node.up.node_id]

            branch_lens_dict.append([[node.node_id], br_len])

        branch_lengths = tf_common.scatter_nd(
                branch_lens_dict,
                output_shape=self.branch_len_inners.shape,
                name="branch_lengths")
        return branch_lengths

    def set_params_from_dict(self, param_dict: Dict[str, ndarray]):
        self.set_params(
                param_dict["target_lams"],
                param_dict["target_lam_decay_rate"],
                param_dict["double_cut_weight"],
                param_dict["boost_softmax_weights"],
                param_dict["trim_long_factor"],
                param_dict["trim_zero_probs"],
                param_dict["trim_short_nbinom_m"],
                param_dict["trim_short_nbinom_logits"],
                param_dict["trim_long_nbinom_m"],
                param_dict["trim_long_nbinom_logits"],
                param_dict["insert_zero_prob"],
                param_dict["insert_nbinom_m"],
                param_dict["insert_nbinom_logit"],
                param_dict["branch_len_inners"],
                param_dict["branch_len_offsets_proportion"],
                param_dict["tot_time_extra"])

    def set_params(
            self,
            target_lams: ndarray,
            target_lam_decay_rate: ndarray,
            double_cut_weight: float,
            boost_softmax_weights: ndarray,
            trim_long_factor: ndarray,
            trim_zero_probs: ndarray,
            trim_short_nbinom_m: ndarray,
            trim_short_nbinom_logits: ndarray,
            trim_long_nbinom_m: ndarray,
            trim_long_nbinom_logits: ndarray,
            insert_zero_prob: ndarray,
            insert_nbinom_m: ndarray,
            insert_nbinom_logit: ndarray,
            branch_len_inners: ndarray,
            branch_len_offsets_proportion: ndarray,
            tot_time_extra: float):
        """
        Set model params
        Should be very similar code to _create_unknown_parameters
        """
        known_vals = np.concatenate([
            [] if not self.known_params.tot_time else [tot_time_extra],
            [] if not self.known_params.target_lams else target_lams,
            [] if not self.known_params.target_lam_decay_rate else target_lam_decay_rate,
            [] if not self.known_params.double_cut_weight else double_cut_weight,
            [] if not self.known_params.trim_long_factor else trim_long_factor,
            [] if not self.known_params.indel_dists else trim_short_nbinom_m,
            [] if not self.known_params.indel_dists else trim_short_nbinom_logits,
            [] if not self.known_params.indel_dists else trim_long_nbinom_m,
            [] if not self.known_params.indel_dists else trim_long_nbinom_logits,
            [] if not self.known_params.indel_dists else insert_nbinom_m,
            [] if not self.known_params.indel_dists else insert_nbinom_logit,
            [] if not self.known_params.indel_params else boost_softmax_weights,
            [] if not self.known_params.indel_params else trim_zero_probs,
            [] if not self.known_params.indel_params else insert_zero_prob,
            branch_len_inners[self.known_params.branch_len_inners] if self.known_params.branch_lens else [],
            branch_len_offsets_proportion[self.known_params.branch_len_offsets_proportion] if self.known_params.branch_lens else []])
        init_val = np.concatenate([
            [] if self.known_params.tot_time else np.log([tot_time_extra]),
            [] if self.known_params.target_lams else np.log(target_lams),
            [] if self.known_params.target_lam_decay_rate else np.log(target_lam_decay_rate),
            [] if self.known_params.double_cut_weight else np.log(double_cut_weight),
            [] if self.known_params.trim_long_factor else inv_sigmoid(trim_long_factor),
            [] if self.known_params.indel_dists else np.log(trim_short_nbinom_m),
            [] if self.known_params.indel_dists else trim_short_nbinom_logits,
            [] if self.known_params.indel_dists else np.log(trim_long_nbinom_m),
            [] if self.known_params.indel_dists else trim_long_nbinom_logits,
            [] if self.known_params.indel_dists else np.log(insert_nbinom_m),
            [] if self.known_params.indel_dists else insert_nbinom_logit,
            [] if self.known_params.indel_params else boost_softmax_weights,
            [] if self.known_params.indel_params else inv_sigmoid(trim_zero_probs),
            [] if self.known_params.indel_params else inv_sigmoid(insert_zero_prob),
            np.log(branch_len_inners[self.known_params.branch_len_inners_unknown] if self.known_params.branch_lens else branch_len_inners),
            inv_sigmoid(branch_len_offsets_proportion[self.known_params.branch_len_offsets_proportion_unknown] if self.known_params.branch_lens else branch_len_offsets_proportion)])

        self.sess.run(
            [self.assign_known_op, self.assign_op],
            feed_dict={
                self.known_vars_ph: known_vals,
                self.all_vars_ph: init_val
            })

    def get_vars(self):
        """
        @return the variable values -- companion for set_params (aka the ordering of the output matches set_params)
        """
        return self.sess.run([
            self.target_lams,
            self.target_lam_decay_rate,
            self.double_cut_weight,
            self.boost_softmax_weights,
            self.trim_long_factor,
            self.trim_zero_probs,
            self.trim_short_nbinom_m,
            self.trim_short_nbinom_logits,
            self.trim_long_nbinom_m,
            self.trim_long_nbinom_logits,
            self.insert_zero_prob,
            self.insert_nbinom_m,
            self.insert_nbinom_logit,
            self.branch_len_inners,
            self.branch_len_offsets_proportion,
            self.tot_time,
            self.tot_time_extra])

    def get_vars_as_dict(self):
        """
        @return the variable values as dictionary instead
        """
        var_vals = self.get_vars()
        var_labels = [
            "target_lams",
            "target_lam_decay_rate",
            "double_cut_weight",
            "boost_softmax_weights",
            "trim_long_factor",
            "trim_zero_probs",
            "trim_short_nbinom_m",
            "trim_short_nbinom_logits",
            "trim_long_nbinom_m",
            "trim_long_nbinom_logits",
            "insert_zero_prob",
            "insert_nbinom_m",
            "insert_nbinom_logit",
            "branch_len_inners",
            "branch_len_offsets_proportion",
            "tot_time",
            "tot_time_extra"]
        assert len(var_labels) == len(var_vals)
        var_dict = {lab: val for lab, val in zip(var_labels, var_vals)}
        return var_dict

    def get_branch_lens(self):
        """
        @return dictionary of branch length (node id to branch length)
        """
        return self.sess.run(self.branch_lens)

    def _are_all_branch_lens_positive(self):
        br_lens = self.get_branch_lens()
        logging.info(self.topology.get_ascii(attributes=["allele_events_list_str"]))
        logging.info(self.topology.get_ascii(attributes=["node_id"]))
        logging.info("br lens %s", br_lens)
        logging.info("where neg %s", np.where(br_lens < 0))
        #logging.info("known inners %s", self.known_params.branch_len_inners)
        #logging.info("known propors %s", self.known_params.branch_len_offsets_proportion)
        #logging.info(self.topology.get_ascii(attributes=["nochad_id"]))
        return np.all(br_lens[1:] > 0)

    def initialize_branch_lens(self, tot_time: float, chronos_lam: float=10, root_unifurc_prop: float= 0.01):
        """
        Initialize branch lengths using chronos estimator by updating the
        model param values in this model (so this function modifies this
        model. doesnt return anything)

        @param tot_time: the total height of the tree
        @param chronos_lam: the penalty param to use with chronos
        """
        # If there are no internal nodes, there is nothing to do here.
        num_internal_nodes = 0
        for node in self.topology.traverse():
            if not node.is_root() and not node.is_leaf():
                num_internal_nodes += 1

        if num_internal_nodes == 0:
            return

        tree = self.topology.copy()
        use_random_assignment = len(tree) > 100

        if not use_random_assignment:
            try:
                # TODO: figure out if there is easy way to use chronos
                # We have this try catch because chronos will sometimes assign zero branch lengths
                is_root_unifurc = len(tree.get_children()) == 1
                chronos_tree = tree.get_children()[0] if is_root_unifurc else tree
                chronos_est = CLTChronosEstimator(
                    chronos_tree,
                    self.bcode_meta,
                    self.scratch_dir,
                    tot_time)
                chronos_tree = chronos_est.estimate(chronos_lam)

                logging.info(chronos_tree.get_ascii(attributes=['dist']))
                # Check that chronos didn't assign terrible branch lengths
                chronos_tree.dist = root_unifurc_prop * tot_time if is_root_unifurc else 0
                chronos_tree.add_feature('dist_to_root', chronos_tree.dist)
                # TODO make this work faster
                for node in chronos_tree.get_descendants('preorder'):
                    if is_root_unifurc:
                        node.dist = (1 - root_unifurc_prop) * node.dist
                    elif node.is_leaf():
                        node.dist = tot_time - node.up.dist_to_root

                    node.add_feature('dist_to_root', node.dist + node.up.dist_to_root)
                    modify_node = node.up if node.is_leaf() else node
                    if node.dist < 1e-8:
                        # Shit chronos screwed up
                        logging.info('chronos assigned bad branch length. reset with random assignments')
                        remain_time = tot_time - node.dist_to_root
                        dist_to_root = 0.95 * remain_time
                        assign_rand_tree_lengths(modify_node, dist_to_root)

                branch_len_inners = np.zeros(self.num_nodes)
                for node in chronos_tree.traverse():
                    branch_len_inners[node.node_id] = node.dist

                # Handle unifurcations in the tree because chronos cannot handle unifurc
                # So let us just split the estimated distance from chronos in half
                for node in self.topology.get_descendants('postorder'):
                    children = node.get_children()
                    if len(children) == 1:
                        child = children[0]
                        orig_dist_assign = branch_len_inners[child.node_id]
                        branch_len_inners[node.node_id] = orig_dist_assign/2
                        branch_len_inners[child.node_id] = orig_dist_assign/2
                logging.info("Chronos branch len inner init %s", branch_len_inners)
                logging.info(np.where(branch_len_inners < 0))
                assert np.all(branch_len_inners[1:] > 0)
            except Exception as e:
                logging.info("Chronos failed")
                use_random_assignment = True

        if use_random_assignment:
            # If chronos fails us, just use random branch lnegth assignments
            assign_rand_tree_lengths(tree, tot_time)
            branch_len_inners = np.zeros(self.num_nodes)
            for node in tree.traverse():
                branch_len_inners[node.node_id] = node.dist

        logging.info("branch len inner init %s", branch_len_inners)
        logging.info(np.where(branch_len_inners < 0))
        assert np.all(branch_len_inners[1:] > 0)

        model_vars = self.get_vars_as_dict()
        model_vars["branch_len_inners"] = branch_len_inners

        # Initialize branch length offsets
        model_vars["branch_len_offsets_proportion"] = np.random.rand(self.num_nodes) * 0.5
        self.set_params_from_dict(model_vars)

        # This line is just to check that the tree is initialized to be ultrametric
        bifurc_tree = self.get_fitted_bifurcating_tree()
        logging.info("init DISTANCE")
        logging.info(bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))

        assert self._are_all_branch_lens_positive()

    def get_all_target_tract_hazards(self):
        return self.sess.run(self.target_tract_hazards)

    def _create_all_target_tract_hazards(self):
        """
        @return Tuple with:
            tensorflow tensor with hazards for all possible target tracts
            dictionary mapping all possible target tracts to their index in the tensor
        """
        target_status_all_active = TargetStatus()
        all_target_tracts = target_status_all_active.get_possible_target_tracts(self.bcode_meta)
        tt_dict = {tt: int(i) for i, tt in enumerate(all_target_tracts)}

        min_targets = tf.constant([tt.min_target for tt in all_target_tracts], dtype=tf.int32)
        max_targets = tf.constant([tt.max_target for tt in all_target_tracts], dtype=tf.int32)
        long_left_statuses = tf.constant([tt.is_left_long for tt in all_target_tracts], dtype=tf.float64)
        long_right_statuses = tf.constant([tt.is_right_long for tt in all_target_tracts], dtype=tf.float64)

        all_hazards = self._create_hazard_target_tract(min_targets, max_targets, long_left_statuses, long_right_statuses)
        return all_hazards, tt_dict

    def _create_hazard_target_tract(
            self,
            min_target: Tensor,
            max_target: Tensor,
            long_left_statuses: Tensor,
            long_right_statuses: Tensor):
        """
        Creates tensorflow node for calculating hazard of a target tract

        @param min_target: the minimum target that was cut
        @param max_target: the maximum target that was cut
        @param long_left_statuses: 1 if long left trim, 0 if short left trim
        @param long_right_statuses: 1 if long right trim, 0 if short left trim

        The arguments should all have the same length.
        The i-th elem in each argument corresponds to the target tract that was introduced.

        @return tensorflow tensor with the i-th value corresponding to the i-th target tract in the arguments
        """
        # Compute the hazard
        # Adding a weight for double cuts for now
        equal_float = tf_common.equal_float(min_target, max_target)
        log_left_trim_factor = tf.log(tf_common.ifelse(long_left_statuses, self.trim_long_factor[0], 1))
        log_right_trim_factor = tf.log(tf_common.ifelse(long_right_statuses, self.trim_long_factor[1], 1))
        log_focal_lambda_part = log_left_trim_factor + log_right_trim_factor + tf.log(tf.gather(self.target_lams, min_target))
        log_double_lambda_part = (log_left_trim_factor
                + tf.log(tf.gather(self.target_lams, min_target) + tf.gather(self.target_lams, max_target))
                + log_right_trim_factor
                + tf.log(self.double_cut_weight))
        hazard = tf.exp(equal_float * log_focal_lambda_part + (1 - equal_float) * log_double_lambda_part, name="hazard")
        return hazard

    def _create_hazard_away_dict(self):
        """
        @return (
            Dictionary mapping all possible TargetStatus to tensorflow tensor for the hazard away,
            Dictionary mapping all possible TargetStatus to tensorflow tensors of hazards of introducing focal and double cuts)
        """
        target_statuses = list(self.targ_stat_transitions_dict.keys())

        # Gets hazards (by tensorflow)
        hazard_away_nodes = self._create_hazard_away_target_statuses(target_statuses)
        hazard_away_dict = {
                targ_stat: hazard_away_nodes[i]
                for i, targ_stat in enumerate(target_statuses)}

        return hazard_away_dict

    def _create_hazard_away_target_statuses(self, target_statuses: List[TargetStatus]):
        """
        @param target_statuses: list of target statuses that we want to calculate the hazard of transitioning away from

        @return tensorflow tensor with the hazards for transitioning away from each of the target statuses
        """
        active_masks = tf.constant(
                [(1 - targ_stat.get_binary_status(self.num_targets)).tolist() for targ_stat in target_statuses],
                dtype=tf.float64)
        active_targ_hazards = self.target_lams * active_masks

        if self.num_targets == 1:
            return active_targ_hazards[:, 0]

        # If more than one target, we need to calculate a lot more things
        focal_hazards = ((1 + self.trim_long_factor[1]) * active_targ_hazards[:, 0]
            + (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * tf.reduce_sum(active_targ_hazards[:, 1:-1], axis=1)
            + (1 + self.trim_long_factor[0]) * active_targ_hazards[:, -1])

        middle_hazards = tf.reduce_sum(active_targ_hazards[:, 1:-1], axis=1)
        num_in_middle = tf.reduce_sum(active_masks[:, 1:-1], axis=1)
        middle_double_cut_tot_haz = (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * (num_in_middle - 1) * middle_hazards

        num_after_start = tf.reduce_sum(active_masks[:, 1:-1], axis=1)
        start_to_other_hazard = active_masks[:,0] * (1 + self.trim_long_factor[1]) * (
                num_after_start * active_targ_hazards[:, 0] + tf.reduce_sum(active_targ_hazards[:, 1:-1], axis=1))

        num_before_end = tf.reduce_sum(active_masks[:, 1:-1], axis=1)
        end_to_other_hazard = active_masks[:,-1] * (1 + self.trim_long_factor[0]) * (
                num_before_end * active_targ_hazards[:, -1] + tf.reduce_sum(active_targ_hazards[:, 1:-1], axis=1))

        start_to_end_hazard = active_masks[:,0] * active_masks[:,-1] * (
                active_targ_hazards[:, 0] + active_targ_hazards[:, -1])

        hazard_away_nodes = focal_hazards + self.double_cut_weight * (
                middle_double_cut_tot_haz
                + start_to_other_hazard
                + end_to_other_hazard
                + start_to_end_hazard)
        return hazard_away_nodes

    """
    SECTION: methods for helping to calculate Pr(indel | target target)
    """
    def _create_log_indel_probs(self, singletons: List[Singleton]):
        """
        Create tensorflow objects for the cond prob of indels

        @return list of tensorflow tensors with indel probs for each singleton
        """
        # TODO: move this check elsewhere? This is to check that all trim lengths make sense!
        for sg in singletons:
            sg.get_trim_lens(self.bcode_meta)

        if not singletons:
            return []
        else:
            insert_probs_boost = self._create_insert_probs(singletons, insert_boost_len=self.boost_len)
            insert_probs = self._create_insert_probs(singletons, insert_boost_len=0)
            left_del_probs_boost = self._create_left_del_probs(singletons, left_boost_len=self.boost_len)
            left_del_probs = self._create_left_del_probs(singletons, left_boost_len=0)
            right_del_probs_boost = self._create_right_del_probs(singletons, right_boost_len=self.boost_len)
            right_del_probs = self._create_right_del_probs(singletons, right_boost_len=0)

            insert_boosted_log_p = tf.log(self.boost_probs[0]) + tf.log(left_del_probs) + tf.log(right_del_probs) + tf.log(insert_probs_boost)
            left_del_boosted_log_p = tf.log(self.boost_probs[1]) + tf.log(left_del_probs_boost) + tf.log(right_del_probs) + tf.log(insert_probs)
            right_del_boosted_log_p = tf.log(self.boost_probs[2]) + tf.log(left_del_probs) + tf.log(right_del_probs_boost) + tf.log(insert_probs)

            # Helps make this numerically stable?
            max_log_p = tf.maximum(tf.maximum(insert_boosted_log_p, left_del_boosted_log_p), right_del_boosted_log_p)
            return max_log_p + tf.log(tf.exp(insert_boosted_log_p - max_log_p) + tf.exp(left_del_boosted_log_p - max_log_p) + tf.exp(right_del_boosted_log_p - max_log_p))

    def _create_left_del_probs(self, singletons: List[Singleton], left_boost_len: int):
        """
        Creates tensorflow nodes that calculate the log conditional probability of the deletions found in
        each of the singletons

        @param left_boost_len: the amount to boost the left short trim

        @return List[tensorflow nodes] for each singleton in `singletons`
        """
        min_targets = [sg.min_target for sg in singletons]
        any_trim_longs = tf.constant(
                [sg.is_left_long or sg.is_right_long for sg in singletons], dtype=tf.float64)
        is_left_longs = tf.constant(
                [sg.is_left_long for sg in singletons], dtype=tf.float64)
        start_posns = tf.constant(
                [sg.start_pos for sg in singletons], dtype=tf.float64)

        # Compute conditional prob of deletion for a singleton
        min_target_sites = tf.constant([self.bcode_meta.abs_cut_sites[mt] for mt in min_targets], dtype=tf.float64)
        left_trim_len = min_target_sites - start_posns

        left_trim_long_min = tf.constant([self.bcode_meta.left_long_trim_min[mt] for mt in min_targets], dtype=tf.float64)
        left_trim_long_max = tf.constant([self.bcode_meta.left_max_trim[mt] for mt in min_targets], dtype=tf.float64)

        min_left_trim = is_left_longs * left_trim_long_min
        max_left_trim = tf_common.ifelse(is_left_longs, left_trim_long_max, left_trim_long_min - 1)

        check_left_max = tf.cast(tf.less_equal(left_trim_len, max_left_trim), tf.float64)
        check_left_min = tf.cast(tf.less_equal(min_left_trim, left_trim_len), tf.float64)
        # The probability of a left trim for length zero in our truncated distribution is assigned to be Pr(0) + Pr(X > max_trim)
        # The other probabilities for the truncated distribution are therefore equal to the original distribution
        short_no_boost_left_prob = tf_common.ifelse(
            tf_common.equal_float(left_trim_len, 0),
            self.trim_zero_prob_left + (1 - self.trim_zero_prob_left) * (
                self.del_short_dist[0].prob(tf.constant(0, dtype=tf.float64)) + tf.constant(1, dtype=tf.float64) - self.del_short_dist[0].cdf(max_left_trim)),
            (1 - self.trim_zero_prob_left) * self.del_short_dist[0].prob(left_trim_len))
        # The truncated distribution for left trims
        num_positions = tf.constant(1.0, dtype=tf.float64) + max_left_trim - min_left_trim
        long_left_prob = self.del_long_dist[0].prob(left_trim_len - min_left_trim) + (tf.constant(1, dtype=tf.float64) - self.del_long_dist[0].cdf(max_left_trim - min_left_trim))/num_positions
        if left_boost_len > 0:
            # If there is a left boost, there are three possibilities:
            # (1) the left trim is smaller than the left boost, so assign probability zero
            # (2) the left trim is equal to the left boost, which means it is zero in the
            # zero-inflated distribution (after we remove the boost)
            # (3) the left trim is longer than the left boost, which is a nonzero value in the
            # zero-inflated distribution (also still must correct for boost)
            short_left_prob = tf_common.ifelse(
                tf_common.less_float(left_trim_len, left_boost_len),
                # Make this nonzero since otherwise the tensorflow gradient calculation will break
                # will be numerically unstable and it will break
                tf.constant(PERTURB_ZERO, dtype=tf.float64),
                self.del_short_dist[0].prob(left_trim_len - left_boost_len))
            return check_left_max * check_left_min * tf_common.ifelse(
                is_left_longs,
                long_left_prob,
                tf_common.ifelse(
                    any_trim_longs,
                    short_no_boost_left_prob,
                    short_left_prob))
        else:
            return check_left_max * check_left_min * tf_common.ifelse(
                is_left_longs,
                long_left_prob,
                short_no_boost_left_prob)

    def _create_right_del_probs(self, singletons: List[Singleton], right_boost_len: int):
        """
        Creates tensorflow nodes that calculate the log conditional probability of the deletions found in
        each of the singletons

        @param right_boost_len: the amount to boost the right short trim

        @return List[tensorflow nodes] for each singleton in `singletons`
        """
        max_targets = [sg.max_target for sg in singletons]
        any_trim_longs = tf.constant(
                [sg.is_left_long or sg.is_right_long for sg in singletons], dtype=tf.float64)
        is_right_longs = tf.constant(
                [sg.is_right_long for sg in singletons], dtype=tf.float64)
        del_ends = tf.constant(
                [sg.del_end for sg in singletons], dtype=tf.float64)

        # Compute conditional prob of deletion for a singleton
        max_target_sites = tf.constant([self.bcode_meta.abs_cut_sites[mt] for mt in max_targets], dtype=tf.float64)
        right_trim_len = del_ends - max_target_sites

        right_trim_long_min = tf.constant([self.bcode_meta.right_long_trim_min[mt] for mt in max_targets], dtype=tf.float64)
        right_trim_long_max = tf.constant([self.bcode_meta.right_max_trim[mt] for mt in max_targets], dtype=tf.float64)

        min_right_trim = is_right_longs * right_trim_long_min
        max_right_trim = tf_common.ifelse(is_right_longs, right_trim_long_max, right_trim_long_min - 1)

        check_right_max = tf.cast(tf.less_equal(right_trim_len, max_right_trim), tf.float64)
        check_right_min = tf.cast(tf.less_equal(min_right_trim, right_trim_len), tf.float64)
        # The probability of a right trim for length zero in our truncated distribution is assigned to be Pr(0) + Pr(X > max_trim)
        # The other probabilities for the truncated distribution are therefore equal to the usual distribution
        short_no_boost_right_prob = tf_common.ifelse(
            tf_common.equal_float(right_trim_len, 0),
            self.trim_zero_prob_right + (1 - self.trim_zero_prob_right) * (
                self.del_short_dist[1].prob(tf.constant(0, dtype=tf.float64)) + tf.constant(1, dtype=tf.float64) - self.del_short_dist[1].cdf(max_right_trim)),
            (1 - self.trim_zero_prob_right) * self.del_short_dist[1].prob(right_trim_len))
        # The truncated distribution for the long right trim (no inflation)
        num_positions = tf.constant(1.0, dtype=tf.float64) + max_right_trim - min_right_trim
        long_right_prob = self.del_long_dist[1].prob(right_trim_len - min_right_trim) + (tf.constant(1, dtype=tf.float64) - self.del_long_dist[1].cdf(max_right_trim - min_right_trim))/num_positions
        if right_boost_len > 0:
            # If there is a right boost, there are three possibilities:
            # (1) the right trim is smaller than the right boost, so assign probability zero
            # (2) the right trim is equal to the right boost, which means it is zero in the
            # zero-inflated distribution (after we remove the boost)
            # (3) the right trim is longer than the right boost, which is a nonzero value in the
            # zero-inflated distribution (also still must correct for boost)
            short_right_prob = tf_common.ifelse(
                tf_common.less_float(right_trim_len, right_boost_len),
                # Make this nonzero since otherwise the tensorflow gradient calculation will break
                # will be numerically unstable and it will break
                tf.constant(PERTURB_ZERO, dtype=tf.float64),
                self.del_short_dist[1].prob(right_trim_len - right_boost_len))
            return check_right_max * check_right_min * tf_common.ifelse(
                is_right_longs,
                long_right_prob,
                tf_common.ifelse(
                    any_trim_longs,
                    short_no_boost_right_prob,
                    short_right_prob))
        else:
            return check_right_max * check_right_min * tf_common.ifelse(
                is_right_longs,
                long_right_prob,
                short_no_boost_right_prob)

    def _create_insert_probs(self, singletons: List[Singleton], insert_boost_len: int):
        """
        Creates tensorflow nodes that calculate the log conditional probability of the insertions found in
        each of the singletons

        @return List[tensorflow nodes] for each singleton in `singletons`
        """
        any_trim_longs = tf.constant(
                [sg.is_left_long or sg.is_right_long for sg in singletons], dtype=tf.float64)
        insert_lens = tf.constant(
                [sg.insert_len for sg in singletons], dtype=tf.float64)
        # Equal prob of all same length sequences
        insert_seq_prob = 1.0/tf.pow(tf.constant(4.0, dtype=tf.float64), insert_lens)
        # If there is no boost, there are two possibilities:
        # (1) the insertion is equal to zero, which means it is zero in the
        # from coin flip or pulled from the usual (neg binom) distribution
        # (2) the insertion is nonzero, which is a nonzero value in the
        # usual distribution
        insert_len_prob = self.insert_dist.prob(insert_lens)
        insert_prob = tf_common.ifelse(
            tf_common.equal_float(insert_lens, 0),
            self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob,
            (1 - self.insert_zero_prob) * insert_len_prob)
        if insert_boost_len > 0:
            # If there is an insertion boost, there are three possibilities:
            # (1) the insertion is smaller than the boost, so assign probability zero
            # (2) the insertion is equal to the boost, which means it is zero in the
            # zero-inflated distribution (after we remove the boost)
            # (3) the insertion is longer than the boost, which is a nonzero value in the
            # zero-inflated distribution (also still must correct for boost)
            # we use maximum to deal with prob assignments outside of support
            insert_boost_len_prob = self.insert_dist.prob(tf.maximum(insert_lens - insert_boost_len, 0))
            insert_boosted_prob = tf_common.ifelse(
                tf_common.less_float(insert_lens, insert_boost_len),
                # Make this nonzero since otherwise the tensorflow gradient calculation will break
                # will be numerically unstable and it will break
                tf.constant(PERTURB_ZERO, dtype=tf.float64),
                insert_boost_len_prob)
                #tf_common.ifelse(
                #    tf_common.equal_float(insert_lens, insert_boost_len),
                #    self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_boost_len_prob,
                #    (1 - self.insert_zero_prob) * insert_boost_len_prob)
            #)
            return tf_common.ifelse(any_trim_longs, insert_prob, insert_boosted_prob) * insert_seq_prob
        else:
            return insert_prob * insert_seq_prob

    """
    LOG LIKELIHOOD CALCULATION section
    """
    @profile
    def create_log_lik(self, transition_wrappers: Dict, create_gradient: bool = True):
        """
        Creates tensorflow nodes that calculate the log likelihood of the observed data
        """
        st_time = time.time()
        self.create_topology_log_lik(transition_wrappers)
        logging.info("Done creating topology log likelihood, time: %d", time.time() - st_time)
        self.log_lik = self.log_lik_alleles

        self.branch_pen = self._create_branch_len_penalties()

        # Penalize target lambdas from being too different
        log_targ_lams = tf.log(self.target_lams)
        self.target_lam_pen = tf.reduce_sum(tf.pow(log_targ_lams - tf.reduce_mean(log_targ_lams), 2))

        # Make our penalized log likelihood
        self.smooth_log_lik = (
                self.log_lik/self.bcode_meta.num_barcodes
                - self.branch_pen * self.branch_pen_param_ph
                - self.target_lam_pen * self.target_lam_pen_param_ph
                - self.crazy_pen_param_ph * tf.reduce_mean(tf.pow(self.all_but_branch_lens, 2)))

        if create_gradient:
            logging.info("Computing gradients....")
            st_time = time.time()
            self.adam_train_op = self.adam_opt.minimize(-self.smooth_log_lik, var_list=self.all_vars)
            logging.info("Finished making me an optimizer, time: %d", time.time() - st_time)

    def _init_singleton_probs(self, singletons: List[Singleton]):
        # Get all the conditional probabilities of the trims
        # Doing it all at once to speed up computation
        self.singleton_index_dict = {sg: int(i) for i, sg in enumerate(singletons)}
        self._create_trim_insert_distributions(len(singletons))
        self.singleton_log_cond_prob = self._create_log_indel_probs(singletons)

    """
    Section for creating the log likelihood of the allele data
    """
    @profile
    def create_topology_log_lik(self, transition_wrappers: Dict[int, List[TransitionWrapper]]):
        """
        Create a tensorflow graph of the likelihood calculation
        """
        singletons = CLTLikelihoodModel.get_all_singletons(self.topology)
        self._init_singleton_probs(singletons)

        # Actually create the nodes for calculating the log likelihoods of the alleles
        self.log_lik_alleles_list = []
        self.Ddiags_list = []
        for bcode_idx in range(self.bcode_meta.num_barcodes):
            print("likelihood bcode", bcode_idx)
            log_lik_alleles, Ddiags = self._create_topology_log_lik_barcode(transition_wrappers, bcode_idx)
            self.log_lik_alleles_list.append(log_lik_alleles)
            self.Ddiags_list.append(Ddiags)
        self.log_lik_alleles = tf.add_n(self.log_lik_alleles_list)

    def _initialize_lower_log_prob(
            self,
            transition_wrapper: TransitionWrapper,
            node: CellLineageTree):
        """
        Initialize the Lprob element with the first part of the product
            For unresolved multifurcs, this is the probability of staying in this same ancestral state (the spine's probability)
                for root nodes, this returns a scalar.
                for non-root nodes, this returns a vector with the initial value for all ancestral states under consideration.
            For resolved multifurcs, this is one
        """
        if not node.is_resolved_multifurcation():
            # Then we need to multiply the probability of the "spine" -- assuming constant ancestral state along the entire spine
            time_stays_constant = tf.reduce_max(tf.stack([
                self.branch_len_offsets[child.node_id]
                for child in node.children]))
            if not node.is_root():
                decay_factor = self._get_decay_factor(
                    self.dist_to_root[node.up.node_id],
                    time_stays_constant)
                # When making this probability, order the elements per the transition matrix of this node
                index_vals = [[
                        [transition_wrapper.key_dict[state], 0],
                        self.hazard_away_dict[state]]
                    for state in transition_wrapper.states]
                haz_aways = tf_common.scatter_nd(
                        index_vals,
                        output_shape=[transition_wrapper.num_possible_states + 1, 1],
                        name="haz_away.multifurc")
                haz_stay_scaled = -haz_aways * decay_factor
                return haz_stay_scaled
            else:
                root_haz_away = self.hazard_away_dict[TargetStatus()]
                decay_factor = self._get_decay_factor(
                    tf.constant(0, dtype=tf.float64),
                    time_stays_constant)
                haz_stay_scaled = -root_haz_away * decay_factor
                return haz_stay_scaled
        else:
            return tf.constant(0, dtype=tf.float64)

    def _get_decay_factor(self, a, delta):
        """
        We suppose a linear decay rate of the instantaneous rate matrix (so a linear decay in the target lam values).
        This is the factor from \int_a^(a + delta) decay(t) dt
        where decay(t) = 1 - c * t/T
        where T = total height of tree

        @return tensorflow node with decay rate
        """
        b = a + delta
        return delta - self.target_lam_decay_rate[0] * (tf.pow(b, 2) - tf.pow(a, 2))/(tf.constant(2, dtype=tf.float64) * self.tot_time)

    @profile
    def _create_branch_len_penalties(self):
        """
        Penalize branch lengths from being too different -- simple ridge penalty to the mean

        @return tensorflow tensor branch length penalty
        """
        branch_lens_to_pen = []
        self.spine_lens = {}
        # Tree traversal order should be postorder
        for node in self.topology.traverse("postorder"):
            if not node.is_leaf():
                if not node.is_resolved_multifurcation():
                    spine_len = tf.reduce_max(tf.stack([
                        self.branch_len_offsets[child.node_id]
                        for child in node.children
                        if not hasattr(child, "ignore_penalty") or not child.ignore_penalty]))
                    branch_lens_to_pen.append(spine_len)
                    self.spine_lens[node.node_id] = spine_len

                for child in node.children:
                    if hasattr(child, "spine_children"):
                        # Only penalize things if there are elements in `spine_children`
                        # Otherwise we should basically ignore this branch lenght penalty
                        if len(child.spine_children):
                            # For a bifurcating tree, this is where we penalize branches and also groups of branches that were originally
                            # a single spine -- we use the instantaneous transition matrix from the top node and multiply by the entire
                            # spine length
                            resolved_spine_len = tf.reduce_sum(
                                    tf.gather(
                                        params=self.branch_lens,
                                        indices=child.spine_children))
                            branch_lens_to_pen.append(resolved_spine_len)
                    elif not hasattr(child, "ignore_penalty") or not child.ignore_penalty:
                        # We sometimes have branches where we need to ignore this penalty
                        # in order to make penalties comparable between different topologies
                        branch_lens_to_pen.append(self.branch_lens[child.node_id])
        log_br = tf.log(branch_lens_to_pen)

        return tf.reduce_mean(tf.pow(log_br - tf.reduce_mean(log_br), 2))

    @profile
    def _create_topology_log_lik_barcode(
            self,
            transition_wrappers: Dict[int, List[TransitionWrapper]],
            bcode_idx: int):
        """
        @param transition_wrappers: dictionary mapping node id to list of TransitionWrapper -- carries useful information
                                    for deciding how to calculate the transition probabilities
        @param bcode_idx: the index of the allele we are calculating the likelihood for

        @return tensorflow tensor with the log likelihood of the allele with index `bcode_idx` for the given tree topology
        """
        # Store the tensorflow objects that calculate the prob of a node being in each state given the leaves
        Lprob = dict()
        Ddiags = dict()
        pt_matrix = dict()
        trans_mats = dict()
        trim_probs = dict()
        down_probs_dict = dict()
        # Store all the scaling terms addressing numerical underflow
        log_scaling_terms = dict()
        # Tree traversal order should be postorder
        for node in self.topology.traverse("postorder"):
            if node.is_leaf():
                node_wrapper = transition_wrappers[node.node_id][bcode_idx]
                prob_array = np.zeros((node_wrapper.num_possible_states + 1, 1))
                observed_key = node_wrapper.key_dict[node_wrapper.leaf_state]
                prob_array[observed_key] = 1
                Lprob[node.node_id] = tf.constant(prob_array, dtype=tf.float64)
            else:
                transition_wrapper = transition_wrappers[node.node_id][bcode_idx]
                log_Lprob_node = self._initialize_lower_log_prob(transition_wrapper, node)

                for child in node.children:
                    child_wrapper = transition_wrappers[child.node_id][bcode_idx]
                    with tf.name_scope("Transition_matrix%d" % node.node_id):
                        trans_mats[child.node_id] = self._create_transition_matrix(
                                child_wrapper)

                    # Get the trim probabilities
                    with tf.name_scope("trim_matrix%d" % node.node_id):
                        trim_probs[child.node_id] = self._create_trim_prob_matrix(child_wrapper)

                    # Create the probability matrix exp(Qt)
                    with tf.name_scope("expm_ops%d" % node.node_id):
                        tr_mat = tf.verify_tensor_all_finite(trans_mats[child.node_id], "transmat %d problem" % child.node_id)
                        decay_factor = self._get_decay_factor(
                            self.dist_to_root[child.node_id] - self.branch_lens[child.node_id],
                            self.branch_lens[child.node_id])
                        pt_matrix[child.node_id], _, _, Ddiags[child.node_id] = tf_common.myexpm(
                                tr_mat,
                                decay_factor)

                    # Get the probability for the data descended from the child node, assuming that the node
                    # has a particular target tract repr.
                    # These down probs are ordered according to the child node's numbering of the TTs states
                    with tf.name_scope("recurse%d" % node.node_id):
                        ch_ordered_down_probs = tf.matmul(
                                tf.multiply(pt_matrix[child.node_id], trim_probs[child.node_id]),
                                Lprob[child.node_id])

                    with tf.name_scope("rearrange%d" % node.node_id):
                        if not node.is_root():
                            # Reorder summands according to node's numbering of tract_repr states
                            node_wrapper = transition_wrappers[node.node_id][bcode_idx]

                            down_probs = CLTLikelihoodModel._reorder_likelihoods(
                                    ch_ordered_down_probs,
                                    node_wrapper,
                                    child_wrapper)
                        else:
                            # For the root node, we just want the probability where the root node is unmodified
                            # No need to reorder
                            ch_id = child_wrapper.key_dict[TargetStatus()]
                            down_probs = ch_ordered_down_probs[ch_id]

                        down_probs_dict[child.node_id] = down_probs
                        if child.is_leaf():
                            leaf_abundance_weight = tf.constant(
                                1 + (child.abundance - 1) * self.abundance_weight,
                                dtype=tf.float64)
                        else:
                            leaf_abundance_weight = tf.constant(1, dtype=tf.float64)
                        log_Lprob_node = log_Lprob_node + tf.log(down_probs) * leaf_abundance_weight

                # Handle numerical underflow
                log_scaling_term = tf.reduce_max(log_Lprob_node)
                Lprob[node.node_id] = tf.verify_tensor_all_finite(
                        tf.exp(log_Lprob_node - log_scaling_term, name="scaled_down_prob"),
                        "lprob%d has problem" % node.node_id)
                log_scaling_terms[node.node_id] = log_scaling_term

        with tf.name_scope("alleles_log_lik"):
            # Account for the scaling terms we used for handling numerical underflow
            log_scaling_terms_all = tf.stack(list(log_scaling_terms.values()))
            log_lik_alleles = tf.add(
                tf.reduce_sum(log_scaling_terms_all, name="add_normalizer"),
                tf.log(Lprob[self.root_node_id]),
                name="alleles_log_lik")

        self.Lprob = Lprob
        self.down_probs_dict = down_probs_dict
        self.pt_matrix = pt_matrix
        self.trans_mats = trans_mats
        self.trim_probs = trim_probs
        return log_lik_alleles, Ddiags

    @profile
    def _create_transition_matrix(self, transition_wrapper: TransitionWrapper):
        """
        @param transition_wrapper: TransitionWrapper that is associated with a particular branch

        @return tensorflow tensor with instantaneous transition rates between meta-states,
                only specifies rates for the meta-states given in `transition_wrapper`.
                So it will create a row for each meta-state in the `transition_wrapper` and then
                create one "impossible" sink state.
                This is the Q matrix for a given branch.
        """
        # Get the target tracts of the singletons -- this is important
        # since the transition matrix excludes the impossible target tracts
        special_tts = set([
                sgwc.get_singleton().get_target_tract()
                for sgwc in transition_wrapper.anc_state.get_singleton_wcs()])

        possible_states = set(transition_wrapper.states)

        single_tt_sparse_indices = []
        single_tt_gather_indices = []
        sparse_indices = []
        sparse_vals = []
        for start_state in transition_wrapper.states:
            start_key = transition_wrapper.key_dict[start_state]
            haz_away = self.hazard_away_dict[start_state]

            # Hazard of staying is negative of hazard away
            sparse_indices.append([start_key, start_key])
            sparse_vals.append(-haz_away)

            all_end_states = set(self.targ_stat_transitions_dict[start_state].keys())
            possible_end_states = all_end_states.intersection(possible_states)
            for end_state in possible_end_states:
                end_key = transition_wrapper.key_dict[end_state]

                # Figure out if this is a special transition involving only a particular target tract
                # Or if it a general any-target-tract transition
                target_tracts_for_transition = self.targ_stat_transitions_dict[start_state][end_state]
                matching_tts = special_tts.intersection(target_tracts_for_transition)
                if matching_tts:
                    assert len(matching_tts) == 1
                    matching_tt = list(matching_tts)[0]
                    single_tt_gather_indices.append(int(self.target_tract_dict[matching_tt]))
                    single_tt_sparse_indices.append([start_key, end_key])
                else:
                    # if we already calculated the hazard of the transition between these target statuses,
                    # use the same node
                    if end_state in self.targ_stat_transition_hazards_dict[start_state]:
                        hazard = self.targ_stat_transition_hazards_dict[start_state][end_state]
                    else:
                        hazard_idxs = [self.target_tract_dict[tt] for tt in target_tracts_for_transition]
                        hazard = tf.reduce_sum(tf.gather(
                            params=self.target_tract_hazards,
                            indices=hazard_idxs))

                        # Store this hazard if we need it in the future
                        self.targ_stat_transition_hazards_dict[start_state][end_state] = hazard

                    sparse_indices.append([start_key, end_key])
                    sparse_vals.append(hazard)

        matrix_len = transition_wrapper.num_possible_states + 1
        if single_tt_gather_indices:
            single_tt_sparse_vals = tf.gather(
                params=self.target_tract_hazards,
                indices=single_tt_gather_indices)
            q_single_tt_matrix = tf.scatter_nd(
                single_tt_sparse_indices,
                single_tt_sparse_vals,
                [matrix_len, matrix_len - 1],
                name="top.q_matrix")
        else:
            q_single_tt_matrix = 0

        # TODO: We might be able to construct this faster using matrix multiplication (or element-wise)
        # rather than this reduce_sum(gather) strategy.
        q_all_tt_matrix = tf.scatter_nd(
            sparse_indices,
            sparse_vals,
            [matrix_len, matrix_len - 1],
            name="top.q_matrix")
        q_matrix = q_all_tt_matrix + q_single_tt_matrix
        # Add hazard to impossible states
        hazard_impossible_states = -tf.reshape(
                tf.reduce_sum(q_matrix, axis=1),
                [matrix_len, 1])

        q_matrix_full = tf.concat([q_matrix, hazard_impossible_states], axis=1)

        return q_matrix_full

    @profile
    def _create_trim_prob_matrix(self, child_transition_wrapper: TransitionWrapper):
        """
        @param transition_wrapper: TransitionWrapper that is associated with a particular branch

        @return matrix of conditional probabilities of each trim
                So the entry in (i,j) is the trim probability for transitioning from target status i
                to target status j. There is a trim prob associated with it because this must correspond to
                the introduction of a singleton in the AncState.

                note: this is used to generate the matrix for a specific branch in the tree
                If the transition is not possible, we fill in with trim prob 1.0 since it doesnt matter
        """
        child_singletons = child_transition_wrapper.anc_state.get_singletons()
        tt_to_singleton = {
                sg.get_target_tract(): sg for sg in child_singletons}
        singleton_target_tracts = set(list(tt_to_singleton.keys()))

        target_status_states = set(child_transition_wrapper.states)
        sparse_indices = []
        singleton_gather_indices = []
        for start_target_status in child_transition_wrapper.states:
            end_state_dict = self.targ_stat_transitions_dict[start_target_status]
            possible_end_target_statuses = set(list(end_state_dict.keys()))
            matching_end_targ_statuses = target_status_states.intersection(possible_end_target_statuses)
            for end_target_status in matching_end_targ_statuses:
                possible_target_tract_transitions = end_state_dict[end_target_status]
                matching_tts = singleton_target_tracts.intersection(possible_target_tract_transitions)
                if matching_tts:
                    assert len(matching_tts) == 1
                    matching_tt = list(matching_tts)[0]
                    matching_singleton = tt_to_singleton[matching_tt]
                    singleton_gather_indices.append(int(self.singleton_index_dict[matching_singleton]))
                    start_key = child_transition_wrapper.key_dict[start_target_status]
                    end_key = child_transition_wrapper.key_dict[end_target_status]
                    sparse_indices.append([start_key, end_key])

        output_length = child_transition_wrapper.num_possible_states + 1
        output_shape = [output_length, output_length]

        if singleton_gather_indices:
            sparse_vals = tf.gather(self.singleton_log_cond_prob, singleton_gather_indices)
            return tf.exp(tf.scatter_nd(
                sparse_indices,
                sparse_vals,
                output_shape,
                name="top.trim_probs"))
        else:
            return tf.ones(output_shape, dtype=tf.float64)

    def get_fitted_bifurcating_tree(self):
        """
        Recall the model was parameterized as a continuous formulation for a multifurcating tree.
        This function returns the bifurcating tree topology using the current model parameters.
        @return CellLineageTree
        """
        # Get the current model parameters
        br_lens, br_len_offsets, tot_time = self.sess.run([
            self.branch_lens,
            self.branch_len_offsets,
            self.tot_time])

        scratch_tree = self.topology.copy("deepcopy")
        for node in scratch_tree.traverse("preorder"):
            node.add_feature("spine_children", [node])

        for node in scratch_tree.traverse("preorder"):
            if not node.is_resolved_multifurcation():
                # Resolve the multifurcation by creating the spine of "identical" nodes
                children = node.get_children()
                children_offsets = [br_len_offsets[c.node_id] for c in children]
                sort_indexes = np.argsort(children_offsets)

                curr_offset = 0
                curr_spine_node = node
                spine_nodes = []
                for idx in sort_indexes:
                    new_spine_node = CellLineageTree(
                            allele_list=node.allele_list,
                            allele_events_list=node.allele_events_list,
                            cell_state=node.cell_state,
                            dist=children_offsets[idx] - curr_offset)
                    curr_spine_node.add_child(new_spine_node)
                    new_spine_node.add_feature("spine_children", [])
                    spine_nodes.append(new_spine_node)

                    child = children[idx]
                    node.remove_child(child)
                    new_spine_node.add_child(child)

                    curr_spine_node = new_spine_node
                    curr_offset = children_offsets[idx]

                node.add_feature("spine_children", [node] + spine_nodes)

            if node.is_root():
                node.dist = 0
            elif len(node.spine_children):
                node.dist = br_lens[node.node_id]

        collapsed_tree._remove_single_child_unobs_nodes(scratch_tree)

        # label node ids but dont override existing leaf node ids
        seen_ids = set([leaf.node_id for leaf in scratch_tree])
        node_id_counter = 0
        tot_nodes = 0
        for node in scratch_tree.traverse():
            tot_nodes += 1
            if not node.is_leaf():
                while node_id_counter in seen_ids:
                    node_id_counter += 1
                node.add_feature("node_id", node_id_counter)
                seen_ids.add(node_id_counter)
            while node_id_counter in seen_ids:
                node_id_counter += 1
        assert node_id_counter == tot_nodes

        remaining_nodes = set([node for node in scratch_tree.traverse()])
        for node in scratch_tree.traverse():
            node.spine_children = [c.node_id for c in node.spine_children if c in remaining_nodes]

        # Just checking that the tree is ultrametric
        for leaf in scratch_tree:
            assert np.isclose(tot_time, leaf.get_distance(scratch_tree))
        return scratch_tree

    @staticmethod
    def get_all_singletons(topology: CellLineageTree):
        singletons = set()
        for leaf in topology:
            for leaf_anc_state in leaf.anc_state_list:
                for singleton_wc in leaf_anc_state.indel_set_list:
                    sg = singleton_wc.get_singleton()
                    singletons.add(sg)
        return singletons

    @staticmethod
    def _reorder_likelihoods(
            ordered_down_probs: Tensor,
            new_wrapper: TransitionWrapper,
            old_wrapper: TransitionWrapper):
        """
        @param ch_ordered_down_probs: the Tensorflow array to be re-ordered
        @param tract_repr_list: list of target tract reprs to include in the vector
                        rest can be set to zero
        @param node_trans_mat: provides the desired ordering
        @param ch_trans_mat: provides the ordering used in vec_lik

        @return the reordered version of vec_lik according to the order in node_trans_mat
        """
        index_vals = [[
                [new_wrapper.key_dict[targ_stat], 0],
                ordered_down_probs[old_wrapper.key_dict[targ_stat]][0]]
            for targ_stat in new_wrapper.states]
        down_probs = tf_common.scatter_nd(
                index_vals,
                output_shape=[new_wrapper.num_possible_states + 1, 1],
                name="top.down_probs")
        return down_probs

    """
    Logger creating/closing functions for debugging
    """
    def create_logger(self):
        self.profile_writer = tf.summary.FileWriter("_output", self.sess.graph)

    def close_logger(self):
        self.profile_writer.close()

    """
    DEBUGGING CODE -- this is for checking the gradient
    """
    def get_log_lik(self, get_grad: bool=False, do_logging: bool=False):
        """
        @return the log likelihood and the gradient, if requested
        """
        if get_grad and not do_logging:
            log_lik, grad = self.sess.run([self.log_lik, self.log_lik_grad])
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

    def check_grad(self, transition_matrices, epsilon=PERTURB_ZERO):
        """
        Function just for checking the gradient
        """
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
