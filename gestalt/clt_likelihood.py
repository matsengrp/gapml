import time
import numpy as np
from typing import List
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix, linalg

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from barcode_events import BarcodeEvents, PlaceholderEvent
from common import product_list

class CLTLikelihoodModel:
    """
    Stores model parameters
    """
    MAX_CUTS = 2
    UNCUT = 0
    REPAIRED = 1
    CUT = 2
    NUM_EVENTS = 3

    def __init__(self, topology: CellLineageTree, num_targets: int):
        self.topology = topology
        self.num_targets = num_targets
        self.random_init()

    def set_vals(self, branch_lens: ndarray, target_lams: ndarray, cell_type_lams: ndarray, repair_lams: ndarray):
        self.branch_lens = branch_lens
        self.target_lams = target_lams
        self.cell_type_lams = cell_type_lams
        self.repair_lams = repair_lams
        assert(self.num_targets == target_lams.size)

    def random_init(self, gamma_prior=(1,10)):
        self.branch_lens = np.random.gamma(gamma_prior[0], gamma_prior[1])
        self.target_lams = np.ones(self.num_targets)
        self.repair_lams = np.ones(self.MAX_CUTS)
        # TODO: implement later!
        self.cell_type_lams = None

    def _create_possible_barcodes(self):
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
        #possible_bcodes = ["".join(bcode) for bcode in possible_bcodes]
        return possible_bcodes

    def create_transition_matrix(self):
        """
        Creates the transition matrix for the given model parameters
        Each target has three possible statuses: 0 - uncut, 1 - repaired with blemishes, 2 - cut and needs repair
        """
        possible_bcodes = self._create_possible_barcodes()

        bcode_row_dict = {"".join([str(t) for t in bcode]): i for i, bcode in enumerate(possible_bcodes)}
        coo_rows = []
        coo_cols = []
        coo_vals = []
        for b_idx, bcode in enumerate(possible_bcodes):
            b_coo_rows = []
            b_coo_cols = []
            b_coo_vals = []
            def _append_to_coo(next_state, rate):
                b_coo_rows.append(b_idx)
                b_coo_cols.append(next_state)
                b_coo_vals.append(rate)

            cut_targets = [i for i, t in enumerate(bcode) if t == self.CUT]
            num_cuts = len(cut_targets)
            if num_cuts < self.MAX_CUTS:
                # Go ahead and cut anywhere
                for t_idx, target in enumerate(bcode):
                    if target == self.UNCUT:
                        next_bcode = bcode[:t_idx] + [self.CUT] + bcode[t_idx + 1:]
                        next_bcode_idx = bcode_row_dict["".join([str(t) for t in next_bcode])]
                        _append_to_coo(next_bcode_idx, self.target_lams[t_idx])

            if num_cuts == 1:
                # focal repair
                rep_idx = cut_targets[0]
                next_bcode = bcode[:rep_idx] + [self.REPAIRED] + bcode[rep_idx + 1:]
                next_bcode_idx = bcode_row_dict["".join([str(t) for t in next_bcode])]
                _append_to_coo(b_idx, self.repair_lams[0])
            elif num_cuts == self.MAX_CUTS:
                # intertarget repair
                rep_idx_start = min(cut_targets)
                rep_idx_end = max(cut_targets)
                num_repairs = rep_idx_end - rep_idx_start + 1
                next_bcode = bcode[:rep_idx_start] + [self.REPAIRED] * num_repairs + bcode[rep_idx_end + 1:]
                next_bcode_idx = bcode_row_dict["".join([str(t) for t in next_bcode])]
                _append_to_coo(next_bcode_idx, self.repair_lams[0])

            assert(num_cuts <= self.MAX_CUTS)

            # transition to self rate
            tot_transition_rate = sum(b_coo_vals)
            _append_to_coo(b_idx, -tot_transition_rate)

            # append to coo
            coo_rows += b_coo_rows
            coo_cols += b_coo_cols
            coo_vals += b_coo_vals

        transition_matrix = coo_matrix((coo_vals, (coo_rows, coo_cols)))
        return transition_matrix.tocsr()


class CLTCalculations:
    """
    Stores parameters useful for likelihood/gradient calculations
    """
    def __init__(self, dl_dbranch_lens: ndarray, dl_dtarget_lams: ndarray, dl_dcell_type_lams: ndarray):
        self.dl_dbranch_lens = dl_dbranch_lens
        self.dl_dtarget_lams = dl_dtarget_lams
        self.dl_dcell_type_lams = dl_dcell_type_lams

class CLTLassoEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        obs_data: List[BarcodeEvents],
        penalty_param: float,
        model_params: CLTLikelihoodModel):
        """
        @param obs_data: observed data
        @param penalty_param: lasso penalty parameter
        @param model_params: initial CLT model params
        """
        self.obs_data = obs_data
        self.penalty_param = penalty_param
        self.model_params = model_params
        self.num_targets = model_params.num_targets

    def get_likelihood(self, model_params: CLTLikelihoodModel, get_grad: bool = False):
        """
        @return The likelihood for proposed theta, the gradient too if requested
        """
        self._get_parsimony_states(model_params)
        self._get_bcode_likelihood(model_params)

    def _get_bcode_likelihood(self, model_params):
        """
        calculates likelihood of just the barcode section
        """
        trans_mat = model_params.create_transition_matrix()

       # # ete takes care of tree traversal - yay no recursion?
       # for node in tree.traverse("postorder"):

    def _get_parsimony_events(self, events1, events2):
        def _is_nested(nester_evt, nestee_evt):
            return (nester_evt.start_pos <= nestee_evt.start_pos
                and nestee_evt.del_end <= nestee_evt.del_end
                and (
                    nester_evt.min_target < nestee_evt.min_target
                    or nestee_evt.max_target < nester_evt.max_target
            ))

        parsimony_evts = [[] for i in range(self.num_targets)]
        all_evts = []
        num_evts = 0
        for idx in range(self.num_targets):
            if len(events1.target_evts[idx]) and len(events2.target_evts[idx]):
                e1 = events1.uniq_events[events1.target_evts[idx][0]]
                e2 = events2.uniq_events[events2.target_evts[idx][0]]
                new_evt = None
                if e1.is_equal(e2):
                    new_evt = e1
                elif not (e1.is_focal and e2.is_focal):
                    if _is_nested(e1, e2):
                        # Most parsimonious thing would be to reuse events
                        new_evt = e2
                    elif _is_nested(e2, e1):
                        # Most parsimonious thing would be to reuse events
                        new_evt = e1
                    else:
                        # No nesting, so events must have arisen separately on the
                        # outgoing branches
                        new_evt = PlaceholderEvent(is_focal=False, target=idx)
                else:
                    # both are focal and are separate events.
                    # must have arisen from an unmodified target
                    continue
                last_evt = parsimony_evts[idx - 1]
                if len(last_evt) and new_evt.is_equal(all_evts[last_evt[0]]):
                    parsimony_evts[idx].append(num_evts - 1)
                else:
                    parsimony_evts[idx].append(num_evts)
                    num_evts += 1
                    all_evts.append(new_evt)

        # TODO: add organ type?
        return BarcodeEvents(parsimony_evts, all_evts, organ=None)


    def _get_parsimony_states(self, model_params):
        """
        get the most parsimonious states for each node in the tree
        """
        for node in self.model_params.topology.traverse("postorder"):
            if node.is_leaf():
                node.add_feature("parsimony_events", node.barcode_events)
            else:
                # TODO: does this work for more internal nodes?
                children_nodes = node.get_children()
                agg_target_evts = children_nodes[0].parsimony_events #barcode_events
                for c in node.children[1:]:
                     agg_target_evts = self._get_parsimony_events(
                             c.parsimony_events, #c.barcode_events,
                             agg_target_evts)
                node.add_feature("parsimony_events", agg_target_evts)
