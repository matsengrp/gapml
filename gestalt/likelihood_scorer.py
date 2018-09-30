from typing import List, Dict
import numpy as np
import tensorflow as tf
import logging
import copy

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from parallel_worker import ParallelWorker
from transition_wrapper_maker import TransitionWrapperMaker
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_estimator import CLTPenalizedEstimator
from model_assessor import ModelAssessor
from optim_settings import KnownModelParams


class LikelihoodScorerResult:
    """
    Stores results from LikelihoodScorer below
    """
    def __init__(
            self,
            fit_params: Dict,
            model_params_dict: Dict,
            orig_tree: CellLineageTree,
            fitted_bifurc_tree: CellLineageTree,
            train_history: List):
        """
        @param fit_params: the fitting parameters used for warm-start
        """
        self.fit_params = fit_params
        self.branch_pen_param = fit_params['branch_pen_param']
        self.target_lam_pen_param = fit_params['target_lam_pen_param']

        self.model_params_dict = model_params_dict
        self.orig_tree = orig_tree
        self.fitted_bifurc_tree = fitted_bifurc_tree
        self.train_history = train_history
        self.pen_log_lik = train_history[-1]["pen_log_lik"]
        self.log_lik = train_history[-1]["log_lik"]

    def get_fit_params(self):
        fit_params = copy.deepcopy(self.model_params_dict)
        fit_params["branch_pen_param"] = self.branch_pen_param
        fit_params["target_lam_pen_param"] = self.target_lam_pen_param
        return fit_params

    def get_all_target_params(self):
        return np.concatenate([
            self.model_params_dict["target_lams"],
            self.model_params_dict["double_cut_weight"],
            self.model_params_dict["trim_long_factor"]])


class LikelihoodScorer(ParallelWorker):
    """
    Fits model parameters and branch lengths for a given tree
    Since this is a parallel worker, it may be used through the job management system SLURM
    """
    def __init__(
            self,
            seed: int,
            tree: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            max_iters: int,
            num_inits: int,
            transition_wrap_maker: TransitionWrapperMaker,
            fit_param_list: List[Dict],
            known_params: KnownModelParams,
            scratch_dir: str,
            assessor: ModelAssessor = None,
            max_try_per_init: int = 2,
            name: str = "likelihoodscorer"):
        """
        @param seed: required to set the seed of each parallel worker
        @param tree: the cell lineage tree topology to fit the likelihood for
        @param bcode_meta: BarcodeMetadata
        @param max_iters: maximum number of iterations for MLE
        @param transition_wrap_maker: TransitionWrapperMaker
        @param fit_param_list: a list of dictionaries specifying model parameter initializations as well
                                    as penalty parameter values. At the very least, each model param list
                                    must contain the penalty parameter settings (branch_pen_param and target lam).
                                    If dictionaries for indices >= 1 have few model param initialization values,
                                    we copy the fitted values over from the previous optimization results. This
                                    serves as a way to warm start.
        @param assessor: if not None, ModelAssessor is used to measure the distance between the estimated
                                tree and the oracle tree at each iteration
        """
        self.seed = seed
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.transition_wrap_maker = transition_wrap_maker
        self.fit_param_list = fit_param_list
        self.known_params = known_params
        self.scratch_dir = scratch_dir
        self.assessor = assessor
        self.max_tries = max_try_per_init * num_inits
        self.name = name

    def run_worker(self, shared_obj=None):
        """
        @param shared_obj: ignored
        """
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            return self.do_work_directly(sess)

    def _fit_one_init(
            self,
            estimator: CLTPenalizedEstimator,
            res_model: CLTLikelihoodModel,
            fit_params: Dict,
            conv_thres_default: float = 1e-6):
        """
        Fit single initialization
        """
        # Initialize branch lengths if not provided
        if 'branch_len_inners' not in fit_params or 'branch_len_offsets_proportion' not in fit_params:
            res_model.initialize_branch_lens(fit_params["tot_time"])
        else:
            logging.info(fit_params["branch_len_inners"])
            logging.info(fit_params["branch_len_offsets_proportion"])

        # Fill in the dictionary for initializing model params
        full_fit_params = res_model.get_vars_as_dict()
        for key, val in fit_params.items():
            if key in full_fit_params.keys():
                if key not in ["tot_time", "tot_time_extra"] and val.shape != full_fit_params[key].shape:
                    raise ValueError(
                            "Something went wrong. not same shape for key %s (%s vs %s)" %
                            (key, val.shape, full_fit_params[key].shape))
                full_fit_params[key] = val
        res_model.set_params_from_dict(full_fit_params)

        # Actually fit the model
        train_history = estimator.fit(
                branch_pen_param=fit_params["branch_pen_param"],
                target_lam_pen_param=fit_params["target_lam_pen_param"],
                conv_thres=fit_params["conv_thres"] if "conv_thres" in fit_params else conv_thres_default,
                assessor=self.assessor)
        result = LikelihoodScorerResult(
            fit_params,
            res_model.get_vars_as_dict(),
            res_model.topology,
            res_model.get_fitted_bifurcating_tree(),
            train_history)
        logging.info(result.fitted_bifurc_tree.get_ascii(attributes=["node_id"], show_internal=True))
        logging.info(result.fitted_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
        logging.info(result.fitted_bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        return result

    def _get_best_result(
            self,
            estimator: CLTPenalizedEstimator,
            model: CLTLikelihoodModel,
            fit_params: Dict):
        """
        For the given `fit_params`, performs multiple initializations -- only returns the best one

        @return LikelihoodScorerResult, returns None if all attempts failed
        """
        logging.info(
                "RUNNING branch pen param %f target lam pen param %f",
                fit_params["branch_pen_param"],
                fit_params["target_lam_pen_param"])
        results = []
        for i in range(self.max_tries):
            try:
                # Note that we will be warm-starting from all the model params
                # Except that we re-initialize branch lengths if they are not provided
                # Pretty reasonable assuming the target lambdas are relatively stable?
                result = self._fit_one_init(
                        estimator,
                        model,
                        fit_params)
                logging.info("Initialization %d result: %f", i, result.pen_log_lik)
            except tf.errors.InvalidArgumentError as e:
                logging.info(e)
                continue
            results.append(result)
            if len(results) >= self.num_inits:
                break

        # Pick out the best result
        if len(results):
            best_idx = np.argmax([r.pen_log_lik for r in results])
            return results[best_idx]
        else:
            logging.info("No training attempt worked")
            return None

    def do_work_directly(self, sess):
        """
        Bypasses all the other code for a ParallelWorker
        Used when we aren't submitting jobs
        Supposes a tensorflow session is given already. Does not make a new one

        @param sess: tensorflow session

        @return List[LikelihoodScorerResult]
        """
        np.random.seed(self.seed)
        res_model = CLTLikelihoodModel(
            self.tree,
            self.bcode_meta,
            sess,
            self.known_params,
            scratch_dir=self.scratch_dir,
            # doesnt matter what value is set here for now. will be overridden
            # TODO: remove this line/argument eventually cause it's ugly...
            target_lams=self.fit_param_list[0]['target_lams'])
        estimator = CLTPenalizedEstimator(
            res_model,
            self.transition_wrap_maker,
            self.max_iters)

        # Fit for each fit-param setting
        result_list = []
        for raw_fit_params in self.fit_param_list:
            fit_params = raw_fit_params.copy()
            if len(result_list):
                # Do warm start from previous results if there are model parameters
                # that the init dictionary does not have
                prev_res = result_list[-1]
                if prev_res is not None:
                    for k, v in prev_res.model_params_dict.items():
                        if k not in fit_params:
                            fit_params[k] = v

            res = self._get_best_result(
                    estimator,
                    res_model,
                    fit_params)
            result_list.append(res)
        return result_list
