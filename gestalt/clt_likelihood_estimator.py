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
    gamma_prior = (1,0.2)
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

    def fit(self, penalty_param, num_inits, max_iters, print_iter=100):
        """
        Finds the best model parameters over `num_inits` initializations
        """
        best_pen_log_lik = self._fit(penalty_param, max_iters, print_iter)
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

            pen_log_lik = self._fit(penalty_param, max_iters, print_iter)
            if pen_log_lik > best_pen_log_lik:
                best_vars = self.model.get_vars()
                best_pen_log_lik = pen_log_lik

        self.model.init_params(*best_vars)
        return best_pen_log_lik

    def _fit(self, penalty_param, max_iters, print_iter):
        """
        Fits for a single initialization
        """
        # Run a gradient descent and see what happens
        pen_log_lik = None
        for i in range(max_iters):
            _, log_lik, log_lik_alleles, log_lik_cell_type, pen_log_lik, grad = self.model.sess.run(
                    [
                        self.model.train_op,
                        self.model.log_lik,
                        self.model.log_lik_alleles,
                        self.model.log_lik_cell_type,
                        self.model.pen_log_lik,
                        self.model.pen_log_lik_grad],
                    feed_dict={
                        self.model.pen_param_ph: penalty_param
                    })
            if i % print_iter == print_iter - 1:
                print("iter", i, "pen log lik", pen_log_lik, "log lik", log_lik, "alleles", log_lik_alleles, "cell type", log_lik_cell_type)

        return pen_log_lik

    def create_logger(self):
        self.model.create_logger()

    def close_logger(self):
        self.model.close_logger()
