import time
import numpy as np
from typing import List
from numpy import ndarray

from clt_estimator import CLTEstimator
from barcode_events import Event, BarcodeEvents, PlaceholderEvent
from clt_likelihood_model import CLTLikelihoodModel

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
        raise NotImplementedError()

    def _get_bcode_likelihood(self, model_params: CLTLikelihoodModel):
        """
        calculates likelihood of just the barcode section
        """
        trans_mat = model_params.create_transition_matrix()
        print(trans_mat)

    def _get_parsimony_events(self, bcode_evt1: BarcodeEvents, bcode_evt2: BarcodeEvents):
        """
        @return the most parsimonious set of events between these two barcodes
        """
        def _is_nested(nester_evt: Event, nestee_evt: Event):
            # @returns whether nestee_evt is completely nested inside nester_evt
            return (nester_evt.start_pos <= nestee_evt.start_pos
                and nestee_evt.del_end <= nestee_evt.del_end
                and (
                    nester_evt.min_target < nestee_evt.min_target
                    or nestee_evt.max_target < nester_evt.max_target
            ))

        target_evts = [[] for i in range(self.num_targets)]
        pars_evts = []
        num_evts = 0
        last_evt = None
        for idx in range(self.num_targets):
            # Check that both targets are associated with an event
            if bcode_evt1.target_evts[idx] and bcode_evt2.target_evts[idx]:
                # We assume that each target is associated with only one event
                # TODO: We'll clean this up in the future. The only reason
                # there may be multiple events is that alignment from Aaron says so.
                # The solution one day will be to clean up the Aaron alignment
                e1 = bcode_evt1.uniq_events[bcode_evt1.target_evts[idx][0]]
                e2 = bcode_evt2.uniq_events[bcode_evt2.target_evts[idx][0]]
                new_evt = None
                if e1.is_equal(e2):
                    # The most parsimonious is that the two events are exactly the same
                    # Then there are zero events needed to explain how the events arose
                    # aka parsimony score contribution zero!
                    new_evt = e1
                elif e1.is_focal and e2.is_focal:
                    # If both events focal but are different events, they can't possibly
                    # have arisen from the same focal event. The only explanation possible
                    # is that they arose separately on each branch from an initial unmodified
                    # target.
                    continue
                else:
                    # At least one event is inter-target
                    # They can't be explained using a parsimony score of zero, but a parsimony
                    # score of one is possible if one event is completely nested within another.
                    if _is_nested(e1, e2):
                        # e2 is nested inside e1
                        new_evt = e2
                    elif _is_nested(e2, e1):
                        # e1 is nested inside e2
                        new_evt = e1
                    else:
                        # No nesting, so events must have arisen separately and there is a
                        # parsimony score contribution of two. However there is ambiguity as to
                        # the original state of the intervening targets. It is possible that in
                        # there was an event located at this target overlapped by both events.
                        # To indicate ambiguity, use a place holder event.
                        new_evt = PlaceholderEvent(is_focal=False, target=idx)

                # Add this shared parsimony event to the list!
                if last_evt is not None and new_evt.is_equal(last_evt):
                    # If we have the same shared event as before, just need to add a pointer
                    # from the target list
                    target_evts[idx].append(num_evts - 1)
                else:
                    # This is a completely new event
                    target_evts[idx].append(num_evts)
                    pars_evts.append(new_evt)
                    num_evts += 1
                last_evt = new_evt

        # TODO: add organ type?
        return BarcodeEvents(target_evts, pars_evts, organ=None)


    def _get_parsimony_states(self, model_params: CLTLikelihoodModel):
        """
        get the most parsimonious states for each node in the tree
        """
        for node in self.model_params.topology.traverse("postorder"):
            if node.is_leaf():
                node.add_feature("parsimony_barcode_events", node.barcode_events)
            else:
                children_nodes = node.get_children()
                # Find the most parsimonious barcode to explain the events observed
                # in children
                pars_bcode_evts = children_nodes[0].parsimony_barcode_events
                for c in node.children[1:]:
                     pars_bcode_evts = self._get_parsimony_events(
                             c.parsimony_barcode_events,
                             pars_bcode_evts)
                node.add_feature("parsimony_barcode_events", pars_bcode_evts)

        # print(self.model_params.topology.get_ascii(attributes=["parsimony_barcode_events"], show_internal=True))
