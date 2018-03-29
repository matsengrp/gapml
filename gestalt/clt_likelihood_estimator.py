import time
from tensorflow import Session
import numpy as np
import scipy.linalg
from typing import List, Tuple, Dict
from numpy import ndarray
import tensorflow as tf
import logging

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
from approximator import ApproximatorLB

class CLTPenalizedEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    gamma_prior = (1,0.2)
    def __init__(
        self,
        model: CLTLikelihoodModel,
        approximator: ApproximatorLB,
        log_barr: float):
        """
        @param model: initial CLT model params
        """
        self.model = model
        self.approximator = approximator
        self.log_barr = log_barr

        # Create the skeletons for the transition matrices -- via state sum approximation
        self.transition_mat_wrappers = self.approximator.create_transition_matrix_wrappers(model)
        self.model.create_log_lik(self.transition_mat_wrappers)
        tf.global_variables_initializer().run()

        # Anything after this is just for testing
        #self.create_logger()
        #st_time = time.time()
        #log_lik, log_lik_grad = self.model.get_log_lik(get_grad=True, do_logging=True)
        #print("tim", time.time() - st_time)
        #print("Log lik", log_lik)
        #print("log lik grad", log_lik_grad)

        #self.model.check_grad(self.transition_mat_wrappers)

    def fit(self, num_inits, max_iters, print_iter=50):
        """
        Finds the best model parameters over `num_inits` initializations
        """
        # Before starting, check that branch lengths are all positive
        br_lens = self.model.get_branch_lens()
        assert all([b > 0 for b in br_lens[1:]])

        best_pen_log_lik = self._fit(max_iters, print_iter)
        best_vars = self.model.get_vars()
        for i in range(num_inits - 1):
            # Initialization -- only perturbing branch lengths and target lams for now...
            branch_lens = np.random.gamma(
                    self.gamma_prior[0],
                    self.gamma_prior[1],
                    self.model.num_nodes)
            # initialize the target lambdas with some perturbation to ensure we
            # don't have eigenvalues that are exactly equal
            target_lams = 0.1 * np.ones(self.model.num_targets) + np.random.uniform(size=self.model.num_targets) * 0.1
            self.model.set_params(
                    target_lams = target_lams,
                    trim_long_probs = best_vars[1],
                    trim_zero_prob = best_vars[2],
                    trim_poissons = best_vars[3],
                    insert_zero_prob = best_vars[4],
                    insert_poisson = best_vars[5],
                    branch_lens = branch_lens,
                    cell_type_lams = best_vars[7])

            pen_log_lik = self._fit(max_iters, print_iter)
            if pen_log_lik > best_pen_log_lik:
                best_vars = self.model.get_vars()
                best_pen_log_lik = pen_log_lik

        self.model.set_params(*best_vars)
        return best_pen_log_lik

    def _fit(self, max_iters, print_iter = 1, step_size = 0.01):
        """
        Fits for a single initialization -- runs proximal gradient descent
        """
        for i in range(max_iters):
            _, log_lik, pen_log_lik, log_lik_alleles, log_lik_cell_type = self.model.sess.run(
                    [
                        self.model.adam_train_op,
                        self.model.log_lik,
                        self.model.smooth_log_lik,
                        self.model.log_lik_alleles,
                        self.model.log_lik_cell_type],
                    feed_dict={
                        self.model.log_barr_ph: self.log_barr
                    })
            assert pen_log_lik != -np.inf

            prev_pen_log_lik = pen_log_lik
            if i % print_iter == print_iter - 1:
                logging.info(
                    "iter %d pen log lik %f log lik %f alleles %f cell type %f",
                    i, pen_log_lik, log_lik, log_lik_alleles, log_lik_cell_type)

        return pen_log_lik

    def create_logger(self):
        self.model.create_logger()

    def close_logger(self):
        self.model.close_logger()
