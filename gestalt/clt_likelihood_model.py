import time
import numpy as np
from typing import List, Tuple, Dict
from numpy import ndarray
from scipy.stats import poisson

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import IndelSet, TargetTract, AncState, SingletonWC, Singleton
from transition_matrix import TransitionMatrixWrapper
from common import merge_target_tract_groups
from bounded_poisson import BoundedPoisson

class CLTLikelihoodModel:
    """
    Stores model parameters
    branch_lens: length of all the branches, indexed by node id at the end of the branch
    target_lams: cutting rate for each target
    cell_type_lams: rate of differentiating to a cell type
    """
    UNLIKELY = "unlikely"
    NODE_ORDER = "postorder"

    def __init__(self, topology: CellLineageTree, bcode_meta: BarcodeMetadata):
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
            target_lams = 0.1 * np.ones(self.num_targets),
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
        if self.topology is None:
            raise ValueError("Must initialize topology first for this to work")

        transition_matrices = dict()
        for node in self.topology.traverse(self.NODE_ORDER):
            if not node.is_root():
                trans_mat = self._create_transition_matrix(node)
                transition_matrices[node.node_id] = trans_mat
        return transition_matrices

    def _create_transition_matrix(self, node: CellLineageTree):
        """
        @return the transition matrix for the particular branch ending at `node`
                only contains the states relevant to state_sum
                the rest of the unlikely states are aggregated together
        """
        transition_dict = dict()
        indel_set_list = node.anc_state.indel_set_list
        # Determine the values in the transition matrix by considering all possible states
        # starting at the parent's StateSum.
        # Recurse through all of its children to build out the transition matrix
        for tts in node.up.state_sum.tts_list:
            tts_partition_info = dict()
            tts_partition = self.partition(tts, node.anc_state)
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
            transition_dict: Dict):
        """
        Recursive function for adding transition matrix rows.
        Function will modify transition_dict.
        The rows added will correspond to states that are reachable along the subgraphs
        in tts_partition_info.

        @param tts_partition_info: indicate the subgraphs for each partition and the current node
                                we are at for each subgraph
        @param indel_set_list: the ordered list of indel sets from the node's AncState
        @param transtion_dict: the dictionary to update with values for the transition matrix
                                (possible key types: tuples of target tracts and the string "unlikely")
        """
        matrix_row = dict()
        start_tts = merge_target_tract_groups([
            tts_partition_info[ind_set]["start"] for ind_set in indel_set_list])
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
            del_prob = self._get_cond_prob_singleton_del(singleton)
            insert_prob = self._get_cond_prob_singleton_insert(singleton)
            prob *= del_prob * insert_prob
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

    def _get_cond_prob_singleton_del(self, singleton: Singleton):
        """
        @return the conditional probability of the deletion for the singleton happening
                (given that the target tract occurred)
        """
        left_trim_len = self.bcode_meta.abs_cut_sites[singleton.min_target] - singleton.start_pos
        right_trim_len = singleton.del_end - self.bcode_meta.abs_cut_sites[singleton.max_target] + 1
        left_trim_long_min = self.bcode_meta.left_long_trim_min[singleton.min_target]
        right_trim_long_min = self.bcode_meta.right_long_trim_min[singleton.max_target]
        if singleton.is_left_long:
            min_left_trim = left_trim_long_min
            max_left_trim = self.bcode_meta.left_max_trim[singleton.min_target]
        else:
            min_left_trim = 0
            max_left_trim = left_trim_long_min - 1

        if singleton.is_right_long:
            min_right_trim = right_trim_long_min
            max_right_trim = self.bcode_meta.right_max_trim[singleton.min_target]
        else:
            min_right_trim = 0
            max_right_trim = right_trim_long_min - 1

        left_prob = BoundedPoisson(min_left_trim, max_left_trim, self.trim_poisson_params[0]).pmf(
            left_trim_len)
        right_prob = BoundedPoisson(min_right_trim, max_right_trim, self.trim_poisson_params[1]).pmf(
            right_trim_len)

        # Check if we should add zero inflation
        if not singleton.is_left_long and not singleton.is_right_long:
            if singleton.del_len == 0:
                return self.trim_zero_prob + (1 - self.trim_zero_prob) * left_prob * right_prob
            else:
                return (1 - self.trim_zero_prob) * left_prob * right_prob
        else:
            return left_prob * right_prob

    def _get_cond_prob_singleton_insert(self, singleton: Singleton):
        """
        @return the conditional probability of the insertion for this singleton happening
                (given that the target tract occurred)
        """
        insert_len = singleton.insert_len
        insert_len_prob = poisson(self.insert_poisson_param).pmf(insert_len)
        # There are 4^insert_len insert string to choose from
        # Assuming all insert strings are equally likely
        insert_seq_prob = 1.0/np.power(4, insert_len)

        # Check if we should add zero inflation
        if insert_len == 0:
            return self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob
        else:
            return (1 - self.insert_zero_prob) * insert_len_prob * insert_seq_prob

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
    def partition(tts: Tuple[IndelSet], anc_state: AncState):
        """
        @return split tts according to anc_state: Dict[IndelSet, Tuple[TargetTract]]
        """
        parts = {indel_set : () for indel_set in anc_state.indel_set_list}

        tts_idx = 0
        n_tt = len(tts)
        anc_state_idx = 0
        n_anc_state = len(anc_state.indel_set_list)
        while tts_idx < n_tt and anc_state_idx < n_anc_state:
            cur_tt = tts[tts_idx]
            indel_set = anc_state.indel_set_list[anc_state_idx]

            if cur_tt.max_deact_target < indel_set.min_deact_target:
                tts_idx += 1
                continue
            elif indel_set.max_deact_target < cur_tt.min_deact_target:
                anc_state_idx += 1
                continue

            # Should be overlapping now
            parts[indel_set] = parts[indel_set] + (cur_tt,)
            tts_idx += 1
        return parts
