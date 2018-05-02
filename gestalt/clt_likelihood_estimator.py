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
from tree_distance import TreeDistanceMeasurerAgg

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
        logging.info("Done creating transition matrices")
        self.model.create_log_lik(self.transition_mat_wrappers)
        logging.info("Done creating tensorflow graph")
        tf.global_variables_initializer().run()

        # Anything after this is just for testing
        #self.create_logger()
        #st_time = time.time()
        #log_lik, log_lik_grad = self.model.get_log_lik(get_grad=True, do_logging=True)
        #print("tim", time.time() - st_time)
        #print("Log lik", log_lik)
        #print("log lik grad", log_lik_grad)

        #self.model.check_grad(self.transition_mat_wrappers)

    def fit(self,
            max_iters: int,
            print_iter: int = 50,
            step_size: float = 0.01,
            dist_measurers: TreeDistanceMeasurerAgg = None):
        """
        Finds the best model parameters
        @param max_iters: number of iterations of gradient descent
        @param print_iter: number of iters to wait to print iterim results
        @param step_size: step size for gradient descent
        @param dist_measurer: if available, this is use to measure how close current tree is to the true tree
                            useful to see how progress is being made
        """
        feed_dict = {
                    self.model.log_barr_ph: self.log_barr,
                    self.model.tot_time_ph: self.model.tot_time
                }
        pen_log_lik = self.model.sess.run(self.model.smooth_log_lik, feed_dict=feed_dict)
        train_history = []
        for i in range(max_iters):
            _, log_lik, pen_log_lik, log_barr, log_lik_alleles, log_lik_cell_type = self.model.sess.run(
                    [
                        self.model.adam_train_op,
                        self.model.log_lik,
                        self.model.smooth_log_lik,
                        self.model.branch_log_barr,
                        self.model.log_lik_alleles,
                        self.model.log_lik_cell_type],
                    feed_dict=feed_dict)
            if pen_log_lik == -np.inf:
                raise ValueError("Penalized log lik not finite, failed on iteration %d" % i)

            iter_info = {
                    "iter": i,
                    "pen_ll": pen_log_lik,
                    "log_barr": log_barr,
                    "log_lik": log_lik}
            prev_pen_log_lik = pen_log_lik
            if i % print_iter == (print_iter - 1):
                logging.info(
                    "iter %d pen log lik %f log lik %f log barr %f alleles %f cell type %f",
                    i, pen_log_lik, log_lik, log_barr, log_lik_alleles, log_lik_cell_type)
                if dist_measurers is not None:
                    tree_dist = dist_measurers.get_tree_dists([self.model.get_fitted_bifurcating_tree()])[0]
                    logging.info("iter %d tree dists: %s", i, tree_dist)
                    iter_info["tree_dists"] = tree_dist
            train_history.append(iter_info)

        return train_history

    def create_logger(self):
        self.model.create_logger()

    def close_logger(self):
        self.model.close_logger()
