import time
import numpy as np
from typing import List
from numpy import ndarray

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
import ancestral_events_finder as anc_evt_finder
from approximator import ApproximatorLB

from state_sum import StateSum

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
        penalty_param: float,
        model_params: CLTLikelihoodModel,
        approximator: ApproximatorLB):
        """
        @param penalty_param: lasso penalty parameter
        @param model_params: initial CLT model params
        """
        self.penalty_param = penalty_param
        self.model_params = model_params
        self.approximator = approximator

        # Annotate with ancestral states
        anc_evt_finder.annotate_ancestral_states(model_params.topology, model_params.bcode_meta)
        # Construct transition boolean matrix -- via state sum approximation
        self.approximator.annotate_state_sum_transitions(model_params.topology)


    def get_likelihood(self, model_params: CLTLikelihoodModel, get_grad: bool = False):
        """
        @return The likelihood for proposed theta, the gradient too if requested
        """
        transition_matrices = model_params.create_transition_matrices()

        L = [dict() for _ in range(model_params.num_nonroot_nodes + 1)]
        for node in model_params.topology.traverse("postorder"):
            if not node.is_leaf():
                for child in node.children:
                    for tts in node.state_sum.tts_set:
                        model_params.get_prob_unmasked_trims(
                            child.anc_state,
                            tts)
        raise NotImplementedError()
