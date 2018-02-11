import time
from tensorflow import Session
import numpy as np
import scipy.linalg
from typing import List, Tuple, Dict
from numpy import ndarray
import tensorflow as tf

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
from approximator import ApproximatorLB
from transition_matrix import TransitionMatrixWrapper, TransitionMatrix
from indel_sets import TargetTract, Singleton

from state_sum import StateSum
from common import target_tract_repr_diff

class CLTLassoEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        penalty_param: float,
        model: CLTLikelihoodModel,
        approximator: ApproximatorLB):
        """
        @param penalty_param: lasso penalty parameter
        @param model: initial CLT model params
        """
        self.penalty_param = penalty_param
        self.model = model
        self.approximator = approximator

        # Create the skeletons for the transition matrices -- via state sum approximation
        self.transition_mat_wrappers = self.approximator.create_transition_matrix_wrappers(model)

        self.model.create_topology_log_lik(self.transition_mat_wrappers)
        self.model.create_logger()
        tf.global_variables_initializer().run()
        st_time = time.time()
        log_lik, log_lik_grad = self.model.get_log_lik(get_grad=True, do_logging=True)
        print("tim", time.time() - st_time)

        #self.model.check_grad(self.transition_mat_wrappers)

        # Run a stupid gradient descent and see what happens
        #train_op = model.grad_opt.minimize(-model.log_lik, var_list=self.model.all_vars)
        #for i in range(100):
        #    _, log_lik = model.sess.run([train_op, model.log_lik])
        #    print("train", log_lik)
        #print(self.model.get_vars())
        self.model.close_logger()

