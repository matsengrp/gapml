import numpy as np
import logging
from tensorflow import Session
from numpy import ndarray

from cell_state import CellTypeTree
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_manipulation import search_nearby_trees
from clt_estimator import CLTEstimator

from parallel_worker import BatchSubmissionManager
from likelihood_scorer import LikelihoodScorer
from simulate_common import fit_pen_likelihood
from tree_distance import UnrootRFDistanceMeasurer

class CLTLikelihoodTopologySearcher:
    """
    Search over tree topologies for the cell lineage tree
    via NNI moves.

    For each iteration:
      Considers multiple trees within X number of NNI moves
      For each tree under consideration, optimizes the model to get penalized log likelihood
        (in a distributed manner if requested)
      Then picks the tree with the best penalized log likelihood.
        (Doesn't move if there is no tree with a better pen log lik)
    """
    def __init__(self,
            bcode_meta: BarcodeMetadata,
            cell_type_tree: CellTypeTree,
            know_cell_lams: bool,
            target_lams: ndarray,
            log_barr: float,
            max_inner_iters: int,
            approximator: ApproximatorLB,
            sess: Session,
            scratch_dir: str,
            tot_time: float = 1,
            true_tree: CellLineageTree = None,
            do_distributed: bool = False):
        """
        @param max_inner_iters: number of iters for optimizing model parameters (not topology search iters)
        @param cell_type_tree: set to none if not using cell type tree
        @param target_lams: set to None if this is not known
        """
        self.bcode_meta = bcode_meta
        self.cell_type_tree = cell_type_tree
        self.know_cell_lams = know_cell_lams
        self.target_lams = target_lams
        self.log_barr = log_barr
        self.tot_time = tot_time
        self.max_inner_iters = max_inner_iters
        self.approximator = approximator
        self.sess = sess
        self.scratch_dir = scratch_dir
        self.true_tree = true_tree
        self.do_distributed = do_distributed

        if true_tree is not None:
            self.tree_dist_measurer = UnrootRFDistanceMeasurer(true_tree, None)

    @staticmethod
    def _assign_branch_lens(br_len_dict, tree):
        """
        Assign branch lengths to this current tree
        """
        for node in tree.traverse():
            if not node.is_root():
                node.dist = br_len_dict[node.node_id]

    def search(self,
            curr_tree: CellLineageTree,
            max_iters: int,
            num_nni_restarts: int,
            max_nni_steps: int,
            do_warm_starts: bool = True):
        # Double check that we aren't starting with the true tree
        if self.true_tree is not None:
            unroot_rf = self.tree_dist_measurer.get_dist(curr_tree)
            logging.info("Current tree")
            logging.info(curr_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
            logging.info("True tree")
            logging.info(self.true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
            #assert unroot_rf > 0

        curr_pen_ll, curr_model = fit_pen_likelihood(
                curr_tree,
                self.bcode_meta,
                self.cell_type_tree,
                self.know_cell_lams,
                self.target_lams,
                self.log_barr,
                self.max_inner_iters,
                self.approximator,
                self.sess,
                tot_time = self.tot_time)
        # Assign branch lengths to this current tree
        CLTLikelihoodTopologySearcher._assign_branch_lens(curr_model.get_branch_lens(), curr_tree)
        curr_model_vars = curr_model.get_vars_as_dict()
        # Now start exploring the space with NNI moves
        logging.info("init tree: unroot rf %d, init pen_ll %f", unroot_rf, curr_pen_ll)
        for i in range(max_iters):
            logging.info("iter %d: Current tree....", i)
            logging.info(curr_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

            # Search around with NNI
            nearby_trees = []
            for _ in range(num_nni_restarts):
                nearby_trees += search_nearby_trees(curr_tree, max_search_dist=max_nni_steps)
            # uniq ones please
            nearby_trees = UnrootRFDistanceMeasurer.get_uniq_trees(nearby_trees)

            # Calculate the likelihoods
            worker_list = [
                LikelihoodScorer(
                        i,
                        tree,
                        self.bcode_meta,
                        self.cell_type_tree,
                        self.know_cell_lams,
                        self.target_lams,
                        self.log_barr,
                        self.max_inner_iters,
                        self.approximator,
                        tot_time = self.tot_time,
                        init_model_vars = curr_model_vars if do_warm_starts else None)
                for i, tree in enumerate(nearby_trees)]
            if self.do_distributed and len(worker_list) > 1:
                # Submit jobs to slurm
                batch_manager = BatchSubmissionManager(
                        worker_list=worker_list,
                        shared_obj=None,
                        num_approx_batches=len(worker_list),
                        worker_folder=self.scratch_dir)
                nni_results = batch_manager.run()
            else:
                # Run jobs locally
                nni_results = [worker.do_work_directly(self.sess) for worker in worker_list]

            for tree, res in zip(nearby_trees, nni_results):
                logging.info("considered tree: pen ll %f", res[0])
                logging.info(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

            # Now let's compare their pen log liks
            pen_lls = [r[0] for r in nni_results]
            best_index = np.argmax(pen_lls)
            if pen_lls[best_index] < curr_pen_ll:
                # None of these are better than the current tree.
                logging.info("None of these are better")
                continue

            # We found a tree with higher pen log lik than current tree
            curr_pen_ll = pen_lls[best_index]
            curr_tree = nearby_trees[best_index]
            curr_model_vars = nni_results[best_index][1]
            curr_br_lens = nni_results[best_index][2]
            logging.info("curr tree pen_ll %f", curr_pen_ll)

            # Store info about our best current tree for warm starting later on
            CLTLikelihoodTopologySearcher._assign_branch_lens(curr_br_lens, curr_tree)

            # Calculate RF distance to understand if our hillclimbing is working
            if self.true_tree is not None:
                unroot_rf = self.tree_dist_measurer.get_dist(curr_tree)
                logging.info("curr tree distance unroot %d", unroot_rf)

        logging.info("final tree pen_ll %f", curr_pen_ll)
        logging.info(curr_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        if self.true_tree is not None:
            logging.info("final tree distance, unroot %d", unroot_rf)
        return curr_tree, curr_model_vars
