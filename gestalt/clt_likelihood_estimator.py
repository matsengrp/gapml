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
from transition_wrapper_maker import TransitionWrapperMaker
from tree_distance import TreeDistanceMeasurerAgg
import plot_simulation_common

class CLTPenalizedEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        model: CLTLikelihoodModel,
        transition_wrapper_maker: TransitionWrapperMaker,
        max_iters: int):
        """
        @param model: initial CLT model params
        @param transition_wrapper_maker: TransitionWrapperMaker
        @param max_iters: maximum number of training iterations
        """
        self.model = model
        self.max_iters = max_iters

        # Create the skeletons for the transition matrices -- via state sum approximation
        transition_wrappers = transition_wrapper_maker.create_transition_wrappers()
        logging.info("Done creating transition wrappers")
        self.model.create_log_lik(
                transition_wrappers,
                create_gradient = max_iters > 0)
        logging.info("Done creating tensorflow graph")
        tf.global_variables_initializer().run()

        # Anything after this is just for testing
        #self.create_logger()
        #st_time = time.time()
        #log_lik, log_lik_grad = self.model.get_log_lik(get_grad=True, do_logging=True)
        #print("tim", time.time() - st_time)
        #print("Log lik", log_lik)
        #print("log lik grad", log_lik_grad)

        #self.model.check_grad(transition_wrappers)

    def fit(self,
            log_barr_pen: float,
            dist_to_half_pen: float = 0,
            print_iter: int = 1,
            save_iter: int = 20,
            dist_measurers: TreeDistanceMeasurerAgg = None,
            conv_thres: float = 1e-5,
            min_iters: int = 10):
        """
        Finds the best model parameters
        @param log_barr: penalty parameter for the log barrier function
        @param dist_to_half_pen: penalty parameter for the log target lambda difference from the mean
        @param print_iter: number of iters to wait to print iterim results
        @param dist_measurer: if available, this is use to measure how close current tree is to the true tree
                            useful to see how progress is being made
        """
        feed_dict = {
                    self.model.log_barr_ph: log_barr_pen,
                    self.model.dist_to_half_pen_ph: dist_to_half_pen,
                }

        pen_log_lik, log_lik, branch_lens, dist_to_half_pen = self.model.sess.run(
            [self.model.smooth_log_lik,
                self.model.log_lik,
                self.model.branch_lens,
                self.model.dist_to_half_pen],
            feed_dict=feed_dict)

        prev_pen_log_lik = pen_log_lik[0]
        logging.info("dist pen %f", dist_to_half_pen)
        logging.info("initial penalized log lik %f, unpen log lik %f", pen_log_lik, log_lik)
        print("initial penalized log lik obtained %f" % pen_log_lik)
        assert not np.isnan(pen_log_lik)
        train_history = [{
                    "iter": -1,
                    "log_lik": log_lik,
                    "pen_log_lik": pen_log_lik,
                    "dist_to_half_pen": dist_to_half_pen,
                    "branch_lens": branch_lens}]

        if dist_measurers is not None:
            bifurc_tree = self.model.get_fitted_bifurcating_tree()
            train_history[0]["tree_dists"] = dist_measurers.get_tree_dists([
                plot_simulation_common._get_leaved_result(bifurc_tree)])[0]
            logging.info("initial tree dists: %s", train_history[0]["tree_dists"])

        st_time = time.time()
        for i in range(self.max_iters):
            var_dict = self.model.get_vars_as_dict()
            boost_probs = np.exp(var_dict["boost_softmax_weights"])/np.sum(
                    np.exp(var_dict["boost_softmax_weights"]))
            logging.info("boost softmax prob %s", boost_probs)
            for k, v in var_dict.items():
                if k not in ["branch_len_offsets_proportion", "branch_len_inners", "boost_probs"]:
                    logging.info("%s: %s", k, v)

            _, pen_log_lik, log_lik, ridge_pen, log_barr, branch_lens = self.model.sess.run(
                    [
                        self.model.adam_train_op,
                        self.model.smooth_log_lik,
                        self.model.log_lik,
                        self.model.dist_to_half_pen,
                        self.model.branch_log_barr,
                        self.model.branch_lens],
                    feed_dict=feed_dict)

            iter_info = {
                    "iter": i,
                    "log_barr": log_barr,
                    "dist_to_half_pen": ridge_pen,
                    "log_lik": log_lik,
                    "pen_log_lik": pen_log_lik,
                    "branch_lens": branch_lens,
                    "target_rates": var_dict["target_lams"]}
            if i % print_iter == (print_iter - 1):
                logging.info(
                    "iter %d pen log lik %f log lik %f dist-to-half pen %f log barr %f min branch len %f",
                    i, pen_log_lik, log_lik, ridge_pen, log_barr, np.min(branch_lens[1:]))

            if np.isnan(pen_log_lik):
                logging.info("ERROR: pen log like is nan. branch lengths are negative?")
                break

            if i % save_iter == (save_iter - 1):
                logging.info("iter %d, train time %f", i, time.time() - st_time)
                if dist_measurers is not None:
                    bifurc_tree = self.model.get_fitted_bifurcating_tree()
                    tree_dist = dist_measurers.get_tree_dists([
                        plot_simulation_common._get_leaved_result(bifurc_tree)])[0]
                    logging.info("iter %d tree dists: %s", i, tree_dist)
                    iter_info["tree_dists"] = tree_dist
                    iter_info["var"] = var_dict
            train_history.append(iter_info)

            if i > min_iters and np.abs((prev_pen_log_lik - pen_log_lik[0])/prev_pen_log_lik) < conv_thres:
                # Convergence reached
                logging.info("Convergence reached")
                break
            prev_pen_log_lik = pen_log_lik[0]

        if dist_measurers is not None:
            bifurc_tree = self.model.get_fitted_bifurcating_tree()
            tree_dist = dist_measurers.get_tree_dists([
                plot_simulation_common._get_leaved_result(bifurc_tree)])[0]
            logging.info("last_iter tree dists: %s", tree_dist)

        logging.info("total train time %f", time.time() - st_time)
        return train_history

    def create_logger(self):
        self.model.create_logger()

    def close_logger(self):
        self.model.close_logger()
