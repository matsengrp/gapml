from typing import Dict
from numpy import ndarray
import numpy as np

from cell_lineage_tree import CellLineageTree

class KnownModelParams:
    """
    Specify which model parameters are known
    """
    def __init__(
            self,
            target_lams: bool = False,
            double_cut_weight: bool = False,
            trim_long_factor: bool = False,
            branch_lens: bool = False,
            branch_len_inners: ndarray = None,
            branch_len_offsets_proportion: ndarray = None,
            cell_lambdas: bool = False,
            tot_time: bool = False,
            indel_params: bool = False):
        self.target_lams = target_lams
        self.double_cut_weight = double_cut_weight or target_lams
        self.trim_long_factor = trim_long_factor or target_lams
        self.cell_lambdas = cell_lambdas
        self.tot_time = tot_time
        self.indel_params = indel_params
        self.indel_poissons = indel_params

        self.branch_lens = branch_lens
        if self.branch_lens:
            self.branch_len_inners = branch_len_inners
            self.branch_len_inners_idxs = np.where(branch_len_inners)[0]
            self.branch_len_inners_unknown = np.logical_not(branch_len_inners)
            self.branch_len_inners_unknown_idxs = np.where(self.branch_len_inners_unknown)[0]
            self.branch_len_offsets_proportion = branch_len_offsets_proportion
            self.branch_len_offsets_proportion_idxs = np.where(branch_len_offsets_proportion)[0]
            self.branch_len_offsets_proportion_unknown = np.logical_not(branch_len_offsets_proportion)
            self.branch_len_offsets_proportion_unknown_idxs = np.where(self.branch_len_offsets_proportion_unknown)[0]
