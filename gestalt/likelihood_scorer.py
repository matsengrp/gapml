from typing import List, Tuple, Dict
from numpy import ndarray
import numpy as np
import tensorflow as tf

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree
from barcode_metadata import BarcodeMetadata
from parallel_worker import ParallelWorker
from transition_wrapper_maker import TransitionWrapperMaker
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_estimator import CLTPenalizedEstimator
from tree_distance import TreeDistanceMeasurerAgg


class LikelihoodScorerResult:
    """
    Stores results from LikelihoodScorer below
    """
    def __init__(self, model_params_dict: Dict, fitted_bifurc_tree: CellLineageTree, train_history: List):
        self.model_params_dict = model_params_dict
        self.fitted_bifurc_tree = fitted_bifurc_tree
        self.train_history = train_history
        self.pen_log_lik = train_history[-1]["pen_log_lik"]

class LikelihoodScorer(ParallelWorker):
    """
    Fits model parameters and branch lengths for a given tree
    Since this is a parallel worker, it may be used through the job management system SLURM
    """
    def __init__(self,
            seed: int,
            tree: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            log_barr: float,
            target_lam_pen: float,
            max_iters: int,
            transition_wrap_maker: TransitionWrapperMaker,
            init_model_params: Dict,
            dist_measurers: TreeDistanceMeasurerAgg = None,
            name: str = "likelihood scorer"):
        """
        @param seed: required to set the seed of each parallel worker
        @param tree: the cell lineage tree topology to fit the likelihood for
        @param bcode_meta: BarcodeMetadata
        @param log_barr: log barrier penalty parameter, i.e. how much to scale the penalty
        @param target_lam_pen: penalty parameter for log target lambda difference, i.e. how much to scale the penalty
                                (penalty tries to keep target lambdas the same)
        @param max_iters: maximum number of iterations for MLE
        @param transition_wrap_maker: TransitionWrapperMaker
        @param tot_time: total height of the tree
        @param dist_measurers: if not None, TreeDistanceMeasurerAgg is used to measure the distance between the estimated
                                tree and the oracle tree at each iteration
        """
        self.seed = seed
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.log_barr = log_barr
        self.target_lam_pen = target_lam_pen
        self.max_iters = max_iters
        self.transition_wrap_maker = transition_wrap_maker
        self.init_model_params = init_model_params
        self.dist_measurers = dist_measurers
        self.name = name

    def run_worker(self, shared_obj):
        """
        @param shared_obj: ignored
        """
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            return self.do_work_directly(sess)

    def do_work_directly(self, sess):
        """
        Bypasses all the other code for a ParallelWorker
        Used when we aren't submitting jobs
        Supposes a tensorflow session is given already. Does not make a new one

        @param sess: tensorflow session

        @return LikelihoodScorerResult
        """
        np.random.seed(self.seed)
        res_model = CLTLikelihoodModel(
            self.tree,
            self.bcode_meta,
            sess,
            target_lams = self.init_model_params["target_lams"],
            target_lams_known = False,
            cell_type_tree = None,
            cell_lambdas_known = False)
        estimator = CLTPenalizedEstimator(
                res_model,
                self.transition_wrap_maker,
                self.max_iters,
                self.log_barr,
                self.target_lam_pen)

        # Initialize with parameters such that the branch lengths are positive
        if 'branch_len_inners' not in self.init_model_params or 'branch_len_offsets_proportion' not in self.init_model_params:
            res_model.initialize_branch_lens()
        full_init_model_params = res_model.get_vars_as_dict()
        for key, val in self.init_model_params.items():
            if key != "tot_time" and val.shape != full_init_model_params[key].shape:
                raise ValueError(
                        "Something went wrong. not same shape for key %s (%s vs %s)" %
                        (key, val.shape, full_init_model_params[key].shape))
            full_init_model_params[key] = val
        res_model.set_params_from_dict(full_init_model_params)

        br_lens = res_model.get_branch_lens()[1:]
        if not np.all(br_lens > 0):
            raise ValueError("not all positive %s" % br_lens)
        assert res_model._are_all_branch_lens_positive()

        train_history = estimator.fit(dist_measurers = self.dist_measurers)
        return LikelihoodScorerResult(
                res_model.get_vars_as_dict(),
                res_model.get_fitted_bifurcating_tree(),
                train_history)

    def __str__(self):
        return self.name
