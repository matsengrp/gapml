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


class LikelihoodScorer(ParallelWorker):
    """
    Fits model parameters and branch lengths for a given tree
    Since this is a parallel worker, it may be used through the job management system SLURM
    """
    def __init__(self,
            seed: int,
            tree: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            cell_type_tree: CellTypeTree,
            know_cell_lams: bool,
            target_lams: ndarray,
            log_barr: float,
            max_iters: int,
            transition_wrap_maker: TransitionWrapperMaker,
            tot_time: float,
            dist_measurers: TreeDistanceMeasurerAgg = None):
        """
        @param seed: required to set the seed of each parallel worker
        @param tree: the cell lineage tree topology to fit the likelihood for
        @param bcode_meta: BarcodeMetadata
        @param cell_type_tree: pass this in if the likelihood includes cell type information
        @param know_cell_lams: whether or not to use the cell type transition rates in `cell_type_tree`
        @param target_lams: if this is not None, then we will use these fixed target lambda rates
        @param log_barr: log barrier penalty parameter, i.e. how much to scale the penalty
        @param max_iters: maximum number of iterations for MLE
        @param transition_wrap_maker: TransitionWrapperMaker
        @param tot_time: total height of the tree
        @param dist_measurers: if not None, TreeDistanceMeasurerAgg is used to measure the distance between the estimated
                                tree and the oracle tree at each iteration
        """
        self.seed = seed
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.cell_type_tree = cell_type_tree
        self.know_cell_lams = know_cell_lams
        self.target_lams = target_lams
        self.log_barr = log_barr
        self.max_iters = max_iters
        self.transition_wrap_maker = transition_wrap_maker
        self.tot_time = tot_time
        self.dist_measurers = dist_measurers

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
        target_lams_known = self.target_lams is not None
        init_target_lams = self.target_lams
        if not target_lams_known:
            # TODO: magic numbers
            init_target_lams = 0.04 * np.ones(self.bcode_meta.n_targets) + np.random.uniform(size=self.bcode_meta.n_targets) * 0.02

        self.tree.label_node_ids()
        res_model = CLTLikelihoodModel(
            self.tree,
            self.bcode_meta,
            sess,
            target_lams = init_target_lams,
            target_lams_known = target_lams_known,
            cell_type_tree = self.cell_type_tree,
            cell_lambdas_known = self.know_cell_lams,
            tot_time = self.tot_time)
        estimator = CLTPenalizedEstimator(
                res_model,
                self.transition_wrap_maker,
                self.log_barr)

        # Initialize with parameters such that the branch lengths are positive
        res_model.initialize_branch_lens()

        train_history = estimator.fit(self.max_iters, dist_measurers = self.dist_measurers)
        return LikelihoodScorerResult(
                res_model.get_vars_as_dict(),
                res_model.get_fitted_bifurcating_tree(),
                train_history)

    def __str__(self):
        return str(self.seed)
