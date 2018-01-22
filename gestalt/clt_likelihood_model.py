import time
import numpy as np
from typing import List, Tuple, Dict
from numpy import ndarray

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract, AncState, SingletonWC, Singleton
from approximator import ApproximatorLB
from transition_matrix import TransitionMatrixWrapper
from common import merge_target_tract_groups, get_bounded_poisson_prob

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type

    TODO: write tests!!!!
    """
    UNLIKELY = "unlikely"
    NODE_ORDER = "postorder"

    def __init__(self, topology: CellLineageTree, bcode_meta: BarcodeMetadata):
        """
        @param topology: provides a topology only (ignore any branch lengths in this tree)
        Will randomly initialize model parameters
        """
        self.topology = topology
        node_id = 0
        for node in topology.traverse(self.NODE_ORDER):
            node.add_feature("node_id", node_id)
            if node.is_root():
                self.root_node_id = node_id
            node_id += 1
        self.num_nodes = node_id
        self.bcode_meta = bcode_meta
        self.num_targets = bcode_meta.n_targets
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
        Sets model parameters

        @param trim_long_probs: [prob of long trim on left, prob of long trim on right]
        @param trim_poisson_params: [poisson param for left trim, poisson param for right trim]
        """
        self.branch_lens = branch_lens
        self.target_lams = target_lams
        self.trim_long_probs = trim_long_probs
        self.trim_long_left = trim_long_probs[0]
        self.trim_long_right = trim_long_probs[1]
        self.trim_zero_prob = trim_zero_prob
        self.trim_poisson_params = trim_poisson_params
        self.insert_zero_prob = insert_zero_prob
        self.insert_poisson_param = insert_poisson_param
        self.cell_type_lams = cell_type_lams
        assert(self.num_targets == target_lams.size)

    def random_init(self, gamma_prior: Tuple[float, float] = (1,10)):
        """
        Randomly initialize model parameters
        """
        self.set_vals(
            # TODO: right now this is initialized to have one extra branch lenght that is ignored
            #       probably want to clean this up later?
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
        @return dictionary of matrices mapping node id to matrix

        TODO: maybe return a list instead of dictionary
        """
        transition_matrices = dict()
        for node in self.topology.traverse(self.NODE_ORDER):
            if not node.is_root():
                # TODO: this should not be node.anc if statesum used a larger number
                ref_anc = node.up
                trans_mat = self._create_transition_matrix(node, ref_anc)
                transition_matrices[node.node_id] = trans_mat
        return transition_matrices

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
        for tts in ref_anc.state_sum.tts_list:
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

        # Add unlikely state
        transition_dict[self.UNLIKELY] = dict()

        # Create sparse transition matrix given the dictionary representation
        return TransitionMatrixWrapper(transition_dict)

    def _add_transition_dict_row(
            self,
            tts_partition_info: Dict[IndelSet, Dict],
            indel_set_list: List[IndelSet],
            transition_dict):
        """
        Recursive function for adding transition matrix rows
        Function will modify transition_dict

        @param tts_partition_info: indicate the subgraphs for each partition and the current node
                                we are at for each subgraph
        @param indel_set_list: the ordered list of indel sets from the node's AncState
        @param transtion_dict: the dictionary to update with values for the transition matrix
        """
        matrix_row = dict()
        start_tts = merge_target_tract_groups([
            tts_partition_info[ind_set]["start"] for ind_set in indel_set_list])
        assert start_tts not in transition_dict.keys()
        transition_dict[start_tts] = matrix_row

        hazard_to_likely = 0
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
                    hazard = self.get_hazard(child.tt_evt)
                    matrix_row[new_tts] = hazard
                    hazard_to_likely += hazard
                else:
                    raise ValueError("already exists?")

                # Recurse
                if new_tts not in transition_dict.keys():
                    self._add_transition_dict_row(new_tts_part_info, indel_set_list, transition_dict)

        # Calculate hazard to all other states (aka the "unlikely" state)
        hazard_away = self.get_hazard_away(start_tts)
        hazard_to_unlikely = hazard_away - hazard_to_likely
        matrix_row[self.UNLIKELY] = hazard_to_unlikely

        # Add hazard to staying in the same state
        matrix_row[start_tts] = -hazard_away

    def get_hazard(self, tt_evt: TargetTract):
        """
        @param tt_evt: the target tract that is getting introduced
        @return hazard of the event happening
        """
        if tt_evt.min_target == tt_evt.max_target:
            # Focal deletion
            lambda_part = self.target_lams[tt_evt.min_target]
        else:
            # Inter-target deletion
            lambda_part = self.target_lams[tt_evt.min_target] * self.target_lams[tt_evt.max_target]

        left_trim_prob = self.trim_long_left if tt_evt.is_left_long else 1 - self.trim_long_left
        right_trim_prob = self.trim_long_right if tt_evt.is_right_long else 1 - self.trim_long_probs[1]
        return lambda_part * left_trim_prob * right_trim_prob

    def _get_hazard_list(self, tts:Tuple[TargetTract], trim_left: bool, trim_right: bool):
        """
        helper function for making lists of hazard rates

        @param tts: the target tract representation for what allele is already existing.
                    so it tells us which remaining targets are active.
        @param trim_left: whether we are trimming left. if so, we take this into account for hazard rates
        @param trim_right: whether we are trimming right. if so, we take this into account for hazard rates

        @return a list of hazard rates associated with the active targets only!
        """
        trim_short_right = 1 - self.trim_long_right if trim_right else 1
        trim_short_left = 1 - self.trim_long_left if trim_left else 1

        if len(tts) == 0:
            hazard_list = np.concatenate([
                # Short trim left
                [self.target_lams[0] * trim_short_left],
                # Any trim
                self.target_lams[1:-1],
                # Short right trim
                [self.target_lams[-1] * trim_short_right]])
        else:
            if tts[0].min_deact_target > 1:
                hazard_list = np.concatenate([
                    # Short trim left
                    [self.target_lams[0] * trim_short_left],
                    # Any trim
                    self.target_lams[1:tts[0].min_deact_target - 1],
                    # Short right trim
                    [self.target_lams[tts[0].min_deact_target - 1] * trim_short_right]])
            elif tts[0].min_deact_target == 1:
                hazard_list = [self.target_lams[0] * trim_short_left * trim_short_right]
            else:
                hazard_list = []

            for i, tt in enumerate(tts[1:]):
                if tts[i].max_deact_target + 1 < tt.min_deact_target - 1:
                    hazard_list = np.concatenate([
                        hazard_list,
                        # Short left trim
                        [self.target_lams[tts[i].max_deact_target + 1] * trim_short_left],
                        # Any trim
                        self.target_lams[tts[i].max_deact_target + 2:tt.min_deact_target - 1],
                        # Short right trim
                        [self.target_lams[tt.min_deact_target - 1] * trim_short_right]])
                elif tts[i].max_deact_target + 1 == tt.min_deact_target - 1:
                    # Single target, short left and right
                    hazard_list = np.concatenate([
                        hazard_list,
                        [self.target_lams[tts[i].max_deact_target + 1] * trim_short_left * trim_short_right]])

            if tts[-1].max_deact_target < self.num_targets - 2:
                hazard_list = np.concatenate([
                    hazard_list,
                    # Short left trim
                    [self.target_lams[tts[-1].max_deact_target + 1] * trim_short_left],
                    # Any trim
                    self.target_lams[tts[-1].max_deact_target + 2:self.num_targets - 1],
                    # Short trim right
                    [self.target_lams[self.num_targets - 1] * trim_short_right]])
            elif tts[-1].max_deact_target == self.num_targets - 2:
                hazard_list = np.concatenate([
                    hazard_list,
                    # Single target, short left and right trim
                    [self.target_lams[tts[-1].max_deact_target + 1] * trim_short_left * trim_short_right]])
        return hazard_list

    def get_hazard_away(self, tts: Tuple[TargetTract]):
        """
        @param tts: the current target tract representation
        @return the hazard to transitioning away from the current state.
        """
        # First compute sum of hazards for all focal deletions
        focal_hazards = self._get_hazard_list(tts, trim_left=True, trim_right=True)

        # Second, compute sum of hazards of all inter-target deletions
        # Compute the hazard of the left target
        # Ending at target -2 (trim off rightmost target)
        left_hazard = self._get_hazard_list(tts, trim_left=True, trim_right=False)[:-1]

        # Compute the hazard of the right target
        # Starting at target 1 (trim off leftmost target)
        right_hazard = self._get_hazard_list(tts, trim_left=False, trim_right=True)[1:]

        left_cum_hazard = np.cumsum(left_hazard)
        return np.sum(focal_hazards) + np.dot(left_cum_hazard, right_hazard)

    def get_prob_unmasked_trims(self, anc_state: AncState, tts: Tuple[TargetTract]):
        """
        @return probability of the trims for the corresponding singletons in the anc_state
        """
        matching_sgs = CLTLikelihoodModel.get_matching_singletons(anc_state, tts)
        prob = 1
        for singleton in matching_sgs:
            # Calculate probability of that singleton
            sg_prob = self._get_cond_prob_singleton(singleton)
            prob *= sg_prob
        return prob

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
        while sg_idx < n_tts and tts_idx < n_tts:
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

    def _get_cond_prob_singleton(self, singleton: Singleton):
        """
        @return the conditional probability of this singleton happening (given that the target tract occurred)
        """
        left_trim_len = self.bcode_meta.abs_cut_sites[singleton.min_target] - singleton.start_pos
        right_trim_len = singleton.del_end - self.bcode_meta.abs_cut_sites[singleton.max_target]
        left_trim_long_min = self.bcode_meta.left_long_trim_min[singleton.min_deact_target]
        right_trim_long_min = self.bcode_meta.right_long_trim_min[singleton.max_deact_target]
        min_left_trim = left_trim_long_min if singleton.is_left_long else 0
        min_right_trim = right_trim_long_min if singleton.is_right_long else 0
        max_left_trim = self.bcode_meta.left_max_trim[singleton.min_target]
        max_right_trim = self.bcode_meta.right_max_trim[singleton.max_target]

        left_prob = get_bounded_poisson_prob(
            left_trim_len,
            min_val = min_left_trim,
            max_val = max_left_trim,
            poisson_param = self.trim_poisson_params[0])
        right_prob = get_bounded_poisson_prob(
            right_trim_len,
            min_val = min_right_trim,
            max_val = max_right_trim,
            poisson_param = self.trim_poisson_params[1])

        # Check if we should add zero inflation
        if not singleton.is_left_long and not singleton.is_right_long:
            if singleton.del_len == 0:
                return self.trim_zero_prob + (1 - self.trim_zero_prob) * left_prob * right_prob
            else:
                return (1 - self.trim_zero_prob) * left_prob * right_prob
        else:
            return left_prob * right_prob
