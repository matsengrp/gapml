from typing import List, Tuple, Dict
from numpy import ndarray
import tensorflow as tf

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from parallel_worker import ParallelWorker
from simulate_common import fit_pen_likelihood
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
    Runs `fit_pen_likelihood` given a tree. Allows for warm starts
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
            approximator: ApproximatorLB,
            tot_time: float,
            init_model_vars: Dict[str, ndarray] = None,
            dist_measurers: TreeDistanceMeasurerAgg = None,
            aux : Dict  = None):
        """
        @param seed: required to set the seed of each parallel worker
        @param args: arguments that provide settings for how to fit the model
        @param tree: the cell lineage tree topology to fit the likelihood for
        @param init_model_vars: the model variables to initialize with
        @param aux: auxiliary info -- any python dict
        """
        self.seed = seed
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.cell_type_tree = cell_type_tree
        self.know_cell_lams = know_cell_lams
        self.target_lams = target_lams
        self.log_barr = log_barr
        self.max_iters = max_iters
        self.approximator = approximator
        self.tot_time = tot_time
        self.init_model_vars = init_model_vars
        self.dist_measurers = dist_measurers
        self.aux = aux

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
        TODO: clean up this code??? this function is super specific. not sure if thats a good thing
        Suppose a tensorflow session is given already. Do not make a new one
        Used when we aren't submitting jobs
        @param sess: tensorflow session
        """
        train_history, res_model = fit_pen_likelihood(
            self.tree,
            self.bcode_meta,
            self.cell_type_tree,
            self.know_cell_lams,
            self.target_lams,
            self.log_barr,
            self.max_iters,
            self.approximator,
            sess,
            self.tot_time,
            warm_start=self.init_model_vars,
            dist_measurers = self.dist_measurers)
        return LikelihoodScorerResult(
                res_model.get_vars_as_dict(),
                res_model.get_fitted_bifurcating_tree(),
                train_history)

    def __str__(self):
        return str(self.seed)
