from typing import List, Dict
import numpy as np
import scipy.stats
import logging

from cell_lineage_tree import CellLineageTree
from tree_distance import TreeDistanceMeasurerAgg
import collapsed_tree


class ModelAssessor:
    def __init__(
            self,
            ref_param_dict: Dict,
            ref_tree: CellLineageTree,
            tree_measurer_classes: List,
            scratch_dir: str,
            leaf_key: str = "leaf_key"):
        """
        Assesses the tree branch length/topology and the model parameter estimates
        @param ref_tree: the tree we want to measure distances to
        @param ref_param_dict: a dictionary with the true model parameters (see the dict from clt_likelihood_model)
        @param tree_measurer_classes: a list of tree measurer classes we want to use to assess trees
        @param scratch_dir: a scratch directory that we can write silly newick files into
        @param leaf_key: the attribute to use for aligning leaves between trees
        """
        self.ref_tree = ref_tree
        self.ref_collapsed_tree = collapsed_tree.collapse_ultrametric(ref_tree)
        self.ref_collapsed_tree.label_node_ids()
        logging.info("num leaves not collapsed version %d", len(self.ref_tree))
        logging.info("num leaves collapsed version %d", len(self.ref_collapsed_tree))
        logging.info(self.ref_collapsed_tree.get_ascii(attributes=["dist"]))
        logging.info(self.ref_collapsed_tree.get_ascii(attributes=[leaf_key]))

        self.leaf_key = leaf_key
        for leaf in ref_tree:
            assert hasattr(leaf, self.leaf_key)

        self.ref_param_dict = ref_param_dict

        self.tree_measurer_classes = tree_measurer_classes
        self.scratch_dir = scratch_dir
        self.param_compare_funcs = {
            "only_targ": self._compare_only_target_lams,
            "targ": self._compare_target_lams,
            "targ_corr": self._target_lams_corr,
            "double": self._compare_double_cut}

    def _get_full_tree_assessor(self, other_tree):
        # Compare to no collapse tree
        # If the other tree has a different set of leaves, figure out which subset of leaves to compare
        # against in the reference tree
        other_tree_leaf_strs = set([getattr(l, self.leaf_key) for l in other_tree])
        keep_leaf_ids = set()
        for leaf in self.ref_tree:
            if getattr(leaf, self.leaf_key) in other_tree_leaf_strs:
                keep_leaf_ids.add(leaf.node_id)
        assert len(keep_leaf_ids) > 1
        ref_tree_pruned = CellLineageTree.prune_tree(self.ref_tree, keep_leaf_ids)

        # Actually do the comparison
        tree_assessor = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            ref_tree_pruned,
            self.tree_measurer_classes,
            self.scratch_dir,
            self.leaf_key)
        return tree_assessor

    def _get_collapse_tree_assessor(self, other_tree):
        other_tree_leaf_strs = set([getattr(l, self.leaf_key) for l in other_tree])
        # If the other tree has a different set of leaves, figure out which subset of leaves to compare
        # against in the reference tree
        keep_leaf_ids = set()
        num_times_observed = {}
        for leaf in self.ref_collapsed_tree:
            leaf_key_val = getattr(leaf, self.leaf_key)
            if leaf_key_val in other_tree_leaf_strs:
                keep_leaf_ids.add(leaf.node_id)
                if leaf_key_val not in num_times_observed:
                    num_times_observed[leaf_key_val] = 1
                else:
                    num_times_observed[leaf_key_val] += 1

        assert len(keep_leaf_ids) > 1
        ref_collapsed_tree_pruned = CellLineageTree.prune_tree(self.ref_collapsed_tree, keep_leaf_ids)

        # Properly adjust abundances in the collapsed and the comparison tree
        # Remember: the true collapsed tree should have abundance = 1
        for leaf in ref_collapsed_tree_pruned:
            leaf.abundance = 1
        # There might be homoplasy, which would lead to abundance > 1 in the estimated tree
        other_tree_collapse_compare = other_tree.copy()
        for leaf in other_tree_collapse_compare:
            leaf.abundance = num_times_observed[getattr(leaf, self.leaf_key)]

        # Actually do the comparison
        tree_collapse_assessor = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            ref_collapsed_tree_pruned,
            self.tree_measurer_classes,
            self.scratch_dir,
            leaf_key=self.leaf_key)

        return tree_collapse_assessor, other_tree_collapse_compare

    def assess(
                self,
                other_tree: CellLineageTree,
                other_param_dict: Dict = None):
        """
        @param other_tree: the tree we want to assess (comparing to the reference tree)
        @param other_param_dict: the estimated model parameters we want to assess

        Note: is able to compare the `other_tree` if `other_tree` contains a subset of the leaves
        in the reference tree
        """
        dist_dict = {}

        # Compare to no collapse tree
        # If the other tree has a different set of leaves, figure out which subset of leaves to compare
        # against in the reference tree
        tree_assessor = self._get_full_tree_assessor(other_tree)
        full_dist_dict = tree_assessor.get_tree_dists([other_tree])[0]
        # Copy over results
        for k, v in full_dist_dict.items():
            dist_dict["full_%s" % k] = v

        # Compare to collapsed tree
        tree_collapse_assessor, other_tree_collapsed = self._get_collapse_tree_assessor(other_tree)
        collapse_dist_dict = tree_collapse_assessor.get_tree_dists([other_tree_collapsed])[0]
        # Copy over results
        for k, v in collapse_dist_dict.items():
            dist_dict["collapse_%s" % k] = v

        # Calculate the other assessment measures now
        if other_param_dict is not None:
            for compare_key, compare_func in self.param_compare_funcs.items():
                dist_dict[compare_key] = compare_func(other_param_dict)
        return dist_dict

    """
    Functions we use for assessing model parameter estimates
    """

    def _compare_double_cut(self, other_param_dict: Dict):
        def _get_double_cut(param_dict: Dict):
            return param_dict["double_cut_weight"]
        return np.linalg.norm(_get_double_cut(self.ref_param_dict) - _get_double_cut(other_param_dict))

    def _target_lams_corr(self, other_param_dict: Dict):
        def _get_target_lams(param_dict: Dict):
            target_lams = param_dict["target_lams"]
            double_weight = param_dict["double_cut_weight"]
            trim_long = param_dict["trim_long_factor"]
            return np.concatenate([target_lams, double_weight, trim_long])

        def _get_only_target_lams(param_dict: Dict):
            return param_dict["target_lams"]

        return scipy.stats.pearsonr(
                _get_only_target_lams(self.ref_param_dict),
                _get_only_target_lams(other_param_dict))[0]

    def _compare_target_lams(self, other_param_dict: Dict):
        def _get_target_lams(param_dict: Dict):
            target_lams = param_dict["target_lams"]
            double_weight = param_dict["double_cut_weight"]
            trim_long = param_dict["trim_long_factor"]
            return np.concatenate([target_lams, double_weight, trim_long])
        return np.linalg.norm(_get_target_lams(self.ref_param_dict) - _get_target_lams(other_param_dict))

    def _compare_only_target_lams(self, other_param_dict: Dict):
        def _get_only_target_lams(param_dict: Dict):
            return param_dict["target_lams"]
        return np.linalg.norm(_get_only_target_lams(self.ref_param_dict) - _get_only_target_lams(other_param_dict))
