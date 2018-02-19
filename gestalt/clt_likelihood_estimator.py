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

class CLTPenalizedEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        model: CLTLikelihoodModel,
        approximator: ApproximatorLB):
        """
        @param penalty_param: lasso penalty parameter
        @param model: initial CLT model params
        """
        self.model = model
        self.approximator = approximator

        # Create the skeletons for the transition matrices -- via state sum approximation
        self.transition_mat_wrappers = self.approximator.create_transition_matrix_wrappers(model)

        self.model.create_topology_log_lik(self.transition_mat_wrappers)

        tf.global_variables_initializer().run()
        #st_time = time.time()
        #log_lik, log_lik_grad = self.model.get_log_lik(get_grad=True, do_logging=True)
        #print("tim", time.time() - st_time)
        #print("Log lik", log_lik)
        #print("log lik grad", log_lik_grad)

        #self.model.check_grad(self.transition_mat_wrappers)

    def fit(self, penalty_param, max_iters):
        # Run a gradient descent and see what happens
        for i in range(max_iters):
            _, log_lik, pen_log_lik, grad = self.model.sess.run(
                    [self.model.train_op, self.model.log_lik, self.model.pen_log_lik, self.model.pen_log_lik_grad],
                    feed_dict={
                        self.model.pen_param_ph: penalty_param
                    })
            print("iter", i, "neg pen log lik", pen_log_lik, "log lik", log_lik)

        for v in self.model.get_vars():
            print(v)

    def create_logger(self):
        self.model.create_logger()

    def close_logger(self):
        self.model.close_logger()
