from typing import List, Dict
import numpy as np

from cell_lineage_tree import CellLineageTree
from tree_distance import TreeDistanceMeasurerAgg


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
        self.ref_param_dict = ref_param_dict
        self.ref_tree = ref_tree
        self.tree_assessor = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            self.ref_tree,
            n_bcodes,
            tree_measurer_classes,
            scratch_dir)
        self.param_compare_funcs = {
            "only_targ": self._compare_only_target_lams,
            "targ": self._compare_target_lams,
            "double": self._compare_double_cut}

    def assess(self, other_param_dict: Dict, other_tree: CellLineageTree):
        dist_dict = self.tree_assessor.get_tree_dists([other_tree])[0]
        for compare_key, compare_func in self.param_compare_funcs.items():
            dist_dict[compare_key] = compare_func(other_param_dict)
        return dist_dict

    def _compare_double_cut(self, other_param_dict: Dict):
        def _get_double_cut(param_dict: Dict):
            return param_dict["double_cut_weight"]
        return np.linalg.norm(_get_double_cut(self.ref_param_dict) - _get_double_cut(other_param_dict))

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
