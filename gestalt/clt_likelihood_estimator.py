import time
import numpy as np
import tensorflow as tf
import logging
from typing import List

from clt_estimator import CLTEstimator
from clt_likelihood_model import CLTLikelihoodModel
from transition_wrapper_maker import TransitionWrapperMaker
from model_assessor import ModelAssessor


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
                create_gradient=max_iters > 0)
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
            log_barr_pen_param: float,
            branch_pen_param: float = 0,
            target_lam_pen_param: float = 0,
            print_iter: int = 1,
            save_iter: int = 40,
            assessor: ModelAssessor = None,
            conv_thres: float = 1e-4,
            min_iters: int = 20):
        """
        Finds the best model parameters
        @param log_barr: penalty parameter for the log barrier function
        @param branch_pen: penalty parameter for the log target lambda difference from the mean
        @param print_iter: number of iters to wait to print iterim results
        @param dist_measurer: if available, this is use to measure how close current tree is to the true tree
                            useful to see how progress is being made
        """
        feed_dict = {
            self.model.log_barr_pen_param_ph: log_barr_pen_param,
            self.model.branch_pen_param_ph: branch_pen_param,
            self.model.target_lam_pen_param_ph: target_lam_pen_param,
        }
        # Check tree is ultrametric
        bifurc_tree = self.model.get_fitted_bifurcating_tree()
        logging.info("init DISTANCE")
        logging.info(bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))

        # Check branch lengths positive
        assert self.model._are_all_branch_lens_positive()

        pen_log_lik, log_lik, branch_pen, dist_to_roots, target_lam_pen = self.model.sess.run(
            [
                self.model.smooth_log_lik,
                self.model.log_lik,
                self.model.branch_pen,
                self.model.dist_to_root,
                self.model.target_lam_pen,
            ],
            feed_dict=feed_dict)
        var_dict = self.model.get_vars_as_dict()

        logging.info(
                "initial penalized log lik %f, unpen log lik %f, branch pen %f, lambda pen %f",
                pen_log_lik, log_lik, branch_pen, target_lam_pen)
        assert not np.isnan(pen_log_lik)
        train_history = [{
                    "iter": -1,
                    "log_lik": log_lik,
                    "pen_log_lik": pen_log_lik}]
        if assessor is not None:
            bifurc_tree = self.model.get_fitted_bifurcating_tree()
            train_history[0]["performance"] = assessor.assess(var_dict, bifurc_tree)
            logging.info("initial tree dists: %s", train_history[0]["performance"])

        st_time = time.time()
        prev_pen_log_lik = pen_log_lik[0]
        for i in range(self.max_iters):
            var_dict = self.model.get_vars_as_dict()
            boost_probs = np.exp(var_dict["boost_softmax_weights"])/np.sum(
                    np.exp(var_dict["boost_softmax_weights"]))
            logging.info("boost softmax prob %s", boost_probs)
            for k, v in var_dict.items():
                if k not in ["branch_len_offsets_proportion", "branch_len_inners", "boost_probs"]:
                    logging.info("%s: %s", k, v)

            _, pen_log_lik, log_lik, branch_pen, target_lam_pen, dist_to_roots = self.model.sess.run(
                    [
                        self.model.adam_train_op,
                        self.model.smooth_log_lik,
                        self.model.log_lik,
                        self.model.branch_pen,
                        self.model.target_lam_pen,
                        self.model.dist_to_root],
                    feed_dict=feed_dict)

            iter_info = {
                    "iter": i,
                    "branch_pen": branch_pen,
                    "target_lam_pen": target_lam_pen,
                    "log_lik": log_lik,
                    "pen_log_lik": pen_log_lik,
                    "target_rates": var_dict["target_lams"],
            }
            if i % print_iter == (print_iter - 1):
                logging.info(
                    "iter %d pen log lik %f log lik %f branch pen %f, lambda pen %f",
                    i, pen_log_lik, log_lik, branch_pen, target_lam_pen)

            if np.isnan(pen_log_lik):
                logging.info("ERROR: pen log like is nan. branch lengths are negative?")
                break

            if i % save_iter == (save_iter - 1):
                iter_info["var_dict"] = var_dict
                iter_info["dist_to_roots"] = dist_to_roots
                logging.info("iter %d, train time %f", i, time.time() - st_time)
                if assessor is not None:
                    bifurc_tree = self.model.get_fitted_bifurcating_tree()
                    logging.info(bifurc_tree.get_ascii(attributes=["dist"]))
                    performance_dict = assessor.assess(var_dict, bifurc_tree)
                    logging.info("iter %d assess: %s", i, performance_dict)
                    iter_info["performance"] = performance_dict

            train_history.append(iter_info)
            if i > min_iters and (pen_log_lik[0] - prev_pen_log_lik)/np.abs(prev_pen_log_lik) < conv_thres:
                # Convergence reached
                logging.info("Convergence reached %f", conv_thres)
                break
            prev_pen_log_lik = pen_log_lik[0]

        train_history[-1]["var_dict"] = var_dict
        train_history[-1]["dist_to_roots"] = dist_to_roots
        if assessor is not None:
            bifurc_tree = self.model.get_fitted_bifurcating_tree()
            performance_dict = assessor.assess(var_dict, bifurc_tree)
            train_history[-1]["performance"] = performance_dict
            logging.info("last_iter tree dists: %s", performance_dict)

        logging.info("total train time %f", time.time() - st_time)
        return train_history
