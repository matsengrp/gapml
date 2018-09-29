from typing import List, Dict
import numpy as np
import scipy.stats

from cell_lineage_tree import CellLineageTree
from tree_distance import TreeDistanceMeasurerAgg
import collapsed_tree


class ModelAssessor:
    def __init__(
            self,
            ref_param_dict: Dict,
            ref_tree: CellLineageTree,
            n_bcodes: int,
            tree_measurer_classes: List,
            scratch_dir: str):
        """
        Assesses the tree branch length/topology and the model parameter estimates
        """
        for node in ref_tree.traverse():
            node.set_allele_list(node.allele_list.create_truncated_version(n_bcodes))
            node.sync_allele_events_list_str()

        self.ref_tree = ref_tree
        self.ref_param_dict = ref_param_dict

        self.ref_collapsed_tree = collapsed_tree.collapse_ultrametric(ref_tree)
        self.ref_collapsed_tree.label_node_ids()
        print("num leaves coll", len(self.ref_collapsed_tree))
        print("num leaves nooo coll", len(self.ref_tree))

        self.tree_measurer_classes = tree_measurer_classes
        self.scratch_dir = scratch_dir
        self.param_compare_funcs = {
            "only_targ": self._compare_only_target_lams,
            "targ": self._compare_target_lams,
            "targ_corr": self._target_lams_corr,
            "double": self._compare_double_cut}

    def assess(self, other_param_dict: Dict, other_tree: CellLineageTree):
        """
        Note: is able to compare the `other_tree` if `other_tree` contains a subset of the leaves
        in the reference tree
        """
        dist_dict = {}

        # Compare to no collapse tree
        other_tree_leaf_strs = set([l.allele_events_list_str for l in other_tree])
        keep_leaf_ids = set()
        for leaf in self.ref_tree:
            if leaf.allele_events_list_str in other_tree_leaf_strs:
                keep_leaf_ids.add(leaf.node_id)
        ref_tree_pruned = CellLineageTree.prune_tree(self.ref_tree, keep_leaf_ids)

        tree_assessor = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            ref_tree_pruned,
            self.tree_measurer_classes,
            self.scratch_dir)

        full_dist_dict = tree_assessor.get_tree_dists([other_tree])[0]
        for k, v in full_dist_dict.items():
            dist_dict["full_%s" % k] = v

        # Collapsed version
        num_times_obs = {}
        keep_leaf_ids = set()
        for leaf in self.ref_collapsed_tree:
            if leaf.allele_events_list_str in other_tree_leaf_strs:
                if leaf.allele_events_list_str not in num_times_obs:
                    num_times_obs[leaf.allele_events_list_str] = 1
                else:
                    num_times_obs[leaf.allele_events_list_str] += 1
                keep_leaf_ids.add(leaf.node_id)
        ref_collapsed_tree_pruned = CellLineageTree.prune_tree(self.ref_collapsed_tree, keep_leaf_ids)
        for leaf in ref_collapsed_tree_pruned:
            leaf.abundance = 1
        other_tree_collapse_compare = other_tree.copy()
        for leaf in other_tree_collapse_compare:
            leaf.abundance = num_times_obs[leaf.allele_events_list_str]

        tree_collapse_assessor = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            ref_collapsed_tree_pruned,
            self.tree_measurer_classes,
            self.scratch_dir)

        collapse_dist_dict = tree_collapse_assessor.get_tree_dists([other_tree_collapse_compare])[0]
        for k, v in collapse_dist_dict.items():
            dist_dict["collapse_%s" % k] = v

        if other_param_dict is not None:
            for compare_key, compare_func in self.param_compare_funcs.items():
                dist_dict[compare_key] = compare_func(other_param_dict)
        return dist_dict

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
