import time
import numpy as np
from typing import List
from numpy import ndarray

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
from ancestral_events_finder import AncestralEventsFinder

import state_sum as StateSum

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
        anc_evt_finder: AncestralEventsFinder):
        """
        @param penalty_param: lasso penalty parameter
        @param model_params: initial CLT model params
        """
        self.penalty_param = penalty_param
        self.model_params = model_params
        self.num_targets = model_params.num_targets
        self.anc_evt_finder = anc_evt_finder

        # Annotate with ancestral states
        self.anc_evt_finder.annotate_ancestral_states(model_params.topology)
        # Construct transition boolean matrix -- via state sum approximation
        self.annotate_transition_bool_matrices(model_params.topology)


    def get_likelihood(self, model_params: CLTLikelihoodModel, get_grad: bool = False):
        """
        @return The likelihood for proposed theta, the gradient too if requested
        """
        raise NotImplementedError()

    def annotate_transition_bool_matrices(self, topology: CellLineageTree):
        for node in tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("state_sum", StateSum())
            else:
                approx_ancestor = node.up
                create_branch_transition_bool_matrix(approx_ancestor, node)

    def create_branch_transition_bool_matrix(self, approx_anc: CellLineageTree, node: CellLineageTree):
        par_state_sum = node.up.state_sum
        max_anc_state = node.anc_state
        for tts in par_state_sum:
            # Partition the TTs first according to max_anc_state
            StateSum.partition_tts(tts, max_anc_state)

