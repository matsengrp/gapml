import time
import numpy as np
from typing import List, Tuple
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from common import product_list

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type
    repair_lams: rate of repair (length 2 - focal and inter-target repair rate)
    """
    MAX_CUTS = 2
    UNCUT = 0
    REPAIRED = 1
    CUT = 2
    NUM_EVENTS = 3

    def __init__(self, topology: CellLineageTree, bcode_metadata: BarcodeMetadata):
        """
        @param topology: provides a topology only (ignore any branch lengths in this tree)
        @param num_targets: number of targets in barcode

        Will randomly initialize model parameters
        """
        self.topology = topology
        self.bcode_metadata = bcode_metadata
        self.random_init()

    def set_vals(self, branch_lens: ndarray, target_lams: ndarray, cell_type_lams: ndarray, repair_lams: ndarray):
        self.branch_lens = branch_lens
        self.target_lams = target_lams
        self.cell_type_lams = cell_type_lams
        self.repair_lams = repair_lams
        assert(self.num_targets == target_lams.size)

    def random_init(self, gamma_prior: Tuple[float, float] = (1,10)):
        self.branch_lens = np.random.gamma(gamma_prior[0], gamma_prior[1])
        self.target_lams = np.ones(self.num_targets)
        self.repair_lams = np.ones(self.MAX_CUTS)
        # TODO: implement later!
        self.cell_type_lams = None

    def _create_possible_barcodes(self):
        """
        @return a list of all possible barcode states represented by a 0-1-2 array.
                only includes barcodes with at most MAX_CUTS
                Note that it is NOT important to indicate which type of repair
                occurred at the target. It is only important to know if a target
                is active, is cut and waiting for repair, or cannot be cut or repaired.
        """
        # Add all the binary bcodes
        binary_bcodes = product_list(range(self.CUT), repeat=self.num_targets)

        all_one_cut_bcodes = []
        # Barcodes with one cut
        binary_bcodes_minus1 = product_list(range(self.CUT), repeat=self.num_targets - 1)
        for bcode in binary_bcodes_minus1:
            one_cut_bcodes = [bcode[:i] + [self.CUT] + bcode[i:] for i in range(self.num_targets)]
            all_one_cut_bcodes += one_cut_bcodes

        # Barcodes with 2 cuts
        all_two_cut_bcodes = []
        binary_bcodes_minus2 = product_list(range(self.CUT), repeat=self.num_targets - 2)
        for bcode in binary_bcodes_minus2:
            two_cut_bcodes = [
                bcode[:i] + [self.CUT] + bcode[i:i +j] + [self.CUT] + bcode[i+j:]
                for i in range(self.num_targets)
                for j in range(self.num_targets - 1 - i)
            ]
            all_two_cut_bcodes += two_cut_bcodes

        # Finally we have all possible bcode states
        possible_bcodes = binary_bcodes + all_one_cut_bcodes + all_two_cut_bcodes
        return possible_bcodes

    def create_transition_matrix(self):
        """
        Creates the transition matrix for the given model parameters
        Each target has three possible statuses: 0 - uncut, 1 - repaired with blemishes, 2 - cut and needs repair
        @return sparse CSR matrix
        """
        possible_bcodes = self._create_possible_barcodes()
        bcode_row_dict = {
            "".join([str(t) for t in bcode]): i
            for i, bcode in enumerate(possible_bcodes)}

        def _get_bcode_idx(barcode_trinary: List[int]):
            # Look up barcode index in our dictionary
            return bcode_row_dict["".join([str(t) for t in barcode_trinary])]

        coo_rows = []
        coo_cols = []
        coo_vals = []
        for b_idx, bcode in enumerate(possible_bcodes):
            # Fill in the transition matrix one row at a time
            b_coo_cols = []
            b_coo_vals = []
            def _append_to_coo(next_state: List[int], rate: float):
                # update the list of matrix vals for this row
                b_coo_cols.append(next_state)
                b_coo_vals.append(rate)

            cut_targets = [i for i, t in enumerate(bcode) if t == self.CUT]
            num_cuts = len(cut_targets)
            if num_cuts < self.MAX_CUTS:
                # Go ahead and cut anywhere
                for t_idx, target in enumerate(bcode):
                    if target == self.UNCUT:
                        next_bcode_idx = _get_bcode_idx(
                            bcode[:t_idx] + [self.CUT] + bcode[t_idx + 1:]
                        )
                        _append_to_coo(next_bcode_idx, self.target_lams[t_idx])

            if num_cuts == 1:
                # focal repair
                rep_idx = cut_targets[0]
                next_bcode_idx = _get_bcode_idx(
                    bcode[:rep_idx] + [self.REPAIRED] + bcode[rep_idx + 1:]
                )
                _append_to_coo(b_idx, self.repair_lams[0])
            elif num_cuts == self.MAX_CUTS:
                # intertarget repair
                rep_idx_start = min(cut_targets)
                rep_idx_end = max(cut_targets)
                num_repairs = rep_idx_end - rep_idx_start + 1
                next_bcode_idx = _get_bcode_idx(
                    bcode[:rep_idx_start] + [self.REPAIRED] * num_repairs + bcode[rep_idx_end + 1:]
                )
                _append_to_coo(next_bcode_idx, self.repair_lams[0])

            # transition to self rate
            tot_transition_rate = sum(b_coo_vals)
            _append_to_coo(b_idx, -tot_transition_rate)

            # append to coo
            coo_rows += [b_idx] * len(b_coo_cols)
            coo_cols += b_coo_cols
            coo_vals += b_coo_vals

        transition_matrix = coo_matrix((coo_vals, (coo_rows, coo_cols)))
        return transition_matrix.tocsr()
