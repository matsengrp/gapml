import time
import numpy as np
from typing import List, Tuple, Dict
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract
from approximator import ApproximatorLB
from common import product_list, merge_target_tract_groups

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

    def __init__(self, topology: CellLineageTree, bcode_meta: BarcodeMetadata):
        """
        @param topology: provides a topology only (ignore any branch lengths in this tree)
        Will randomly initialize model parameters
        """
        self.topology = topology
        node_id = 0
        for node in topology.traverse("preorder"):
            node.add_feature("node_id", node_id)
            node_id += 1
        self.num_nodes = node_id
        self.bcode_meta = bcode_meta
        self.num_targets = bcode_meta.num_targets
        self.random_init()

    def set_vals(self,
            branch_lens: ndarray,
            target_lams: ndarray,
            trim_long_probs: ndarray,
            trim_zero_prob: float,
            trim_poisson_params: ndarray,
            insert_zero_prob: float,
            insert_poisson_param: float,
            cell_type_lams: ndarray):
        """
        @param trim_long_probs: [prob of long trim on left, prob of long trim on right]
        @param trim_poisson_params: [poisson param for left trim, poisson param for right trim]
        """
        self.branch_lens = branch_lens
        self.target_lams = target_lams
        self.trim_long_probs = trim_long_probs
        self.trim_zero_prob = trim_zero_prob
        self.trim_poisson_params = trim_poisson_params
        self.insert_zero_prob = insert_zero_prob
        self.insert_poisson_param = insert_poisson_param
        self.cell_type_lams = cell_type_lams
        assert(self.num_targets == target_lams.size)

    def random_init(self, gamma_prior: Tuple[float, float] = (1,10)):
        self.set_vals(
            branch_lens = np.random.gamma(gamma_prior[0], gamma_prior[1], self.num_nodes),
            target_lams = 0.5 * np.ones(self.num_targets),
            trim_long_probs = 0.1 * np.ones(2),
            trim_zero_prob = 0.5,
            trim_poisson_params = np.ones(2),
            insert_zero_prob = 0.5,
            insert_poisson_param = 2,
            # TODO: implement later
            cell_type_lams = None,
        )

    def create_transition_matrices(self):
        """
        Create transition matrix for each branch
        """
        for node in self.topology.traverse("preorder"):
            if not node.is_root():
                # TODO: this should not be node.anc if statesum used a larger number
                ref_anc = node.up
                trans_mat = self._create_transition_matrix(node, ref_anc)
                print("node", node.node_id, "mat", trans_mat)
        1/0

    def _create_transition_matrix(self, node: CellLineageTree, ref_anc: CellLineageTree):
        """
        Creates the transition matrix for the particular branch ending at `node`
        @return sparse CSR matrix
        """
        transition_dict = dict()
        indel_set_list = node.anc_state.indel_set_list
        # Determine the values in the transition matrix by considering all possible states
        # starting at the ref_anc's StateSum.
        # Recurse through all of its children to build out the transition matrix
        for tts in ref_anc.state_sum.tts_set:
            tts_partition_info = dict()
            tts_partition = ApproximatorLB.partition(tts, node.anc_state)
            for indel_set in indel_set_list:
                tt_tuple = tts_partition[indel_set]
                graph_key = (tt_tuple, indel_set)
                # To recurse, indicate the subgraphs for each partition and the current node
                # (target tract group) we are currently located at.
                tts_partition_info[indel_set] = {
                        "start": tt_tuple,
                        "graph": node.transition_graph_dict[graph_key]}
            self._add_transition_dict_row(tts_partition_info, indel_set_list, transition_dict)

        if len(transition_dict) == 1:
            return None

        # Create transition matrix given the dictionary representation
        tts_list = []
        tts_dict = dict()
        i = 0
        for tts in transition_dict.keys():
            tts_list.append(tts)
            tts_dict[tts] = i
            i += 1

        coo_rows = []
        coo_cols = []
        coo_vals = []
        for i, tts in enumerate(tts_list):
            for new_tts, val in transition_dict[tts].items():
                coo_rows.append(i)
                coo_cols.append(tts_dict[new_tts])
                coo_vals.append(val)

            # TODO: Add the unlikely state

        transition_matrix = coo_matrix((coo_vals, (coo_rows, coo_cols)))
        return transition_matrix.tocsr()

    def _add_transition_dict_row(
            self,
            tts_partition_info: Dict[IndelSet, Dict],
            indel_set_list: List[IndelSet],
            transition_dict):
        """
        @param tts_partition_info: indicate the subgraphs for each partition and the current node
                                we are at for each subgraph
        @param indel_set_list: the ordered list of indel sets from the node's AncState
        @param transtion_dict: the dictionary to update with values for the transition matrix

        Recursive function for adding transition matrix rows
        Function will modify transition_dict
        """
        matrix_row = dict()
        start_tts = merge_target_tract_groups([
            tts_partition_info[ind_set]["start"] for ind_set in indel_set_list])
        assert start_tts not in transition_dict.keys()
        transition_dict[start_tts] = matrix_row

        # Find all possible target tract representations within one step of start_tts
        # Do this by taking one step in one of the subgraphs
        for indel_set, val in tts_partition_info.items():
            subgraph = val["graph"]
            tt_tuple_start = val["start"]

            # Each child is a possibility
            children = subgraph.get_children(tt_tuple_start)
            for child in children:
                new_tts_part_info = {k: v.copy() for k,v in tts_partition_info.items()}
                new_tts_part_info[indel_set]["start"] = child.tt_group

                # Create the new target tract representation
                new_tts = merge_target_tract_groups([
                    new_tts_part_info[ind_set]["start"] for ind_set in indel_set_list])

                # Add entry to transition matrix
                if new_tts not in matrix_row:
                    # TODO: put a real value here
                    matrix_row[new_tts] = 1
                else:
                    raise ValueError("already exists?")

                # Recurse
                if new_tts not in transition_dict.keys():
                    self._add_transition_dict_row(new_tts_part_info, indel_set_list, transition_dict)
