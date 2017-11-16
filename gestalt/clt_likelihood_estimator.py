import time
import numpy as np
from typing import List
from numpy import ndarray

from clt_estimator import CLTEstimator
from clt_likelihood_model import CLTLikelihoodModel
from ancestral_events_finder import AncestralEventsFinder


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

    def get_likelihood(self, model_params: CLTLikelihoodModel, get_grad: bool = False):
        """
        @return The likelihood for proposed theta, the gradient too if requested
        """
        self.anc_evt_finder.annotate_ancestral_events(model_params.topology)
        self._get_bcode_likelihood(model_params)
        raise NotImplementedError()

    def _get_bcode_likelihood(self, model_params: CLTLikelihoodModel):
        """
        calculates likelihood of just the barcode section
        """
        trans_mat = model_params.create_transition_matrix()
