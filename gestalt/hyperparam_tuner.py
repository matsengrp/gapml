import numpy as np
from typing import List, Dict, Tuple
import logging
import random

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from split_data import create_kfold_trees, create_kfold_barcode_trees, TreeDataSplit
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from parallel_worker import SubprocessManager
from common import get_randint
from model_assessor import ModelAssessor
from optim_settings import KnownModelParams


class PenaltyScorerResult:
    def __init__(
            self,
            score: float,
            fit_results: List[LikelihoodScorerResult]):
        """
        @param score: assumed to be higher the better
        """
        self.score = score
        self.fit_results = fit_results


class PenaltyTuneResult:
    def __init__(
            self,
            tree: CellLineageTree,
            tree_splits: List[TreeDataSplit],
            results: List[PenaltyScorerResult]):
        self.tree = tree
        self.tree_splits = tree_splits
        self.results = results

    def get_best_result(self, warm_idx=0):
        """
        @param warm_idx: the index of the fitted model params to use for warm starts
        @return Dict with model/optimization params
        """
        pen_param_scores = np.array([r.score for r in self.results])
        logging.info("Tuning scores %s", pen_param_scores)
        best_idx = np.argmax(pen_param_scores)
        best_pen_result = self.results[best_idx]

        chosen_best_res = best_pen_result.fit_results[warm_idx]

        fit_params = chosen_best_res.get_fit_params()
        # Popping branch length estimates because we cannot
        # initialize branch lengths from the kfold trees for the full tree
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        return self.tree, fit_params, chosen_best_res


def tune(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor,
        conv_thres: float = 1e-6):
    """
    Tunes the `branch_pen_param`, `target_lam_pen_param` penalty parameters

    @return PenaltyTuneResult
    """
    assert len(args.branch_pen_params) > 1 or len(args.target_lam_pen_params) > 1

    # First create the initialization/optimization settings under consideration
    fit_param_list = []
    for branch_pen_param in args.branch_pen_params:
        for target_lam_pen_param in args.target_lam_pen_params:
            if len(fit_param_list) == 0:
                new_fit_params = fit_params.copy()
            else:
                new_fit_params = {}
            new_fit_params["branch_pen_param"] = branch_pen_param
            new_fit_params["target_lam_pen_param"] = target_lam_pen_param
            new_fit_params["conv_thres"] = conv_thres
            fit_param_list.append(new_fit_params)

    if bcode_meta.num_barcodes > 1:
        # For many barcodes, we split by barcode
        return _tune_hyperparams(
            fit_param_list,
            tree,
            bcode_meta,
            args,
            fit_params,
            create_kfold_barcode_trees,
            _get_many_bcode_hyperparam_score,
            assessor)
    else:
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        # For single barcode, we split into subtrees
        return _tune_hyperparams(
            fit_param_list,
            tree,
            bcode_meta,
            args,
            fit_params,
            create_kfold_trees,
            _get_one_bcode_hyperparam_score,
            assessor)


def _tune_hyperparams(
        fit_param_list: List[Dict],
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        kfold_fnc,
        hyperparam_score_fnc,
        assessor: ModelAssessor = None):
    """
    @param max_num_chad_parents: max number of chad parents to consider
    @param conv_thres: the convergence threshold for training the model
                        (should probably pick something similar to the one being used
                        for the hanging chads)

    @return List[PenaltyScorerResult] -- corresponding to each hyperparam
                being tuned
    """
    # First split the barcode into kfold groups
    n_splits = args.num_penalty_tune_splits if bcode_meta.num_barcodes == 1 else min(args.num_penalty_tune_splits, bcode_meta.num_barcodes)
    logging.info("Hyperparam tuning %d splits", n_splits)

    # Make all the tree splits, but we will only fit a random number of them
    all_tree_splits = kfold_fnc(tree, bcode_meta, n_splits)
    random.shuffle(all_tree_splits)
    tree_splits = all_tree_splits[:args.max_fit_splits]

    trans_wrap_makers = [TransitionWrapperMaker(
            tree_split.train_clt,
            tree_split.train_bcode_meta,
            args.max_extra_steps,
            args.max_sum_states) for tree_split in tree_splits]

    # Actually fit the trees using the kfold barcodes
    worker_list = [LikelihoodScorer(
        get_randint(),
        tree_split.train_clt,
        tree_split.train_bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        fit_param_list=fit_param_list,
        known_params=args.known_params,
        scratch_dir=args.scratch_dir,
        assessor=assessor)
        for tree_split, transition_wrap_maker in zip(tree_splits, trans_wrap_makers)]

    if args.num_processes > 1 and len(worker_list) > 1:
        job_manager = SubprocessManager(
                worker_list,
                None,
                args.scratch_dir,
                args.num_processes)
        train_results = [r for r, _ in job_manager.run()]
    else:
        train_results = [w.run_worker(None) for w in worker_list]
    train_results = [(res, tree_split) for res, tree_split in zip(train_results, tree_splits) if res is not None]

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates and branch length estimates
    tune_results = []
    for idx, fit_param in enumerate(fit_param_list):
        branch_pen_param = fit_param['branch_pen_param']
        target_lam_pen_param = fit_param['target_lam_pen_param']
        logging.info(
                "PEN PARAM setting %d: branch_pen_param %f target_lam_pen_param %f",
                idx,
                fit_param['branch_pen_param'],
                fit_param['target_lam_pen_param'])
        res_folds = [(train_res[idx], tree_split) for train_res, tree_split in train_results]
        hyperparam_score = hyperparam_score_fnc(
            res_folds,
            args.max_extra_steps,
            args.max_sum_states,
            args.scratch_dir,
            args.num_processes)

        # Create our summary of tuning
        tune_result = PenaltyScorerResult(
            hyperparam_score,
            res_folds)
        tune_results.append(tune_result)
        logging.info(
                "Pen param branch %f, target_lam %f, hyperparam score %s",
                branch_pen_param,
                target_lam_pen_param,
                tune_result.score)

    return PenaltyTuneResult(
                tree,
                tree_splits,
                tune_results)


def _get_many_bcode_hyperparam_score(
        pen_param_results: List[Tuple[LikelihoodScorerResult, TreeDataSplit]],
        max_extra_steps: int,
        max_sum_states: int,
        scratch_dir: str,
        num_processes: int):
    """
    @return score = the validation log likelihood
    """
    for pen_param_res, _ in pen_param_results:
        if pen_param_res is None:
            # This pen param setting is not stable
            return -np.inf

    worker_list = []
    for pen_param_res, tree_split in pen_param_results:
        # Use all the fitted params from the training data since we have the
        # same tree topology
        fit_params = pen_param_res.get_fit_params()
        all_known_params = KnownModelParams(
            target_lams=True,
            tot_time=True,
            indel_params=True)
        transition_wrap_maker = TransitionWrapperMaker(
            tree_split.val_clt,
            tree_split.val_bcode_meta,
            max_extra_steps,
            max_sum_states)
        scorer = LikelihoodScorer(
            get_randint(),  # seed
            tree_split.val_clt,
            tree_split.val_bcode_meta,
            # No iterations because we just want to evaluate a probability
            max_iters=0,
            num_inits=1,
            transition_wrap_maker=transition_wrap_maker,
            fit_param_list=[fit_params],
            known_params=all_known_params,
            scratch_dir=scratch_dir)
        worker_list.append(scorer)

    job_manager = SubprocessManager(
            worker_list,
            None,
            scratch_dir,
            num_processes)
    worker_results = [w[0][0] for w in job_manager.run()]
    val_log_liks = [res.log_lik for res in worker_results]

    tot_val_log_lik = np.sum(val_log_liks)
    logging.info("all hyperparam split-scores %s, (sum %f)", val_log_liks, tot_val_log_lik)
    return tot_val_log_lik


def _get_one_bcode_hyperparam_score(
        pen_param_results: List[Tuple[LikelihoodScorerResult, TreeDataSplit]],
        max_extra_steps: int,
        max_sum_states: int,
        scratch_dir: str,
        num_processes: int):
    """
    @return score = Pr(validation data | train data)
    """
    for pen_param_res, _ in pen_param_results:
        if pen_param_res is None:
            # This pen param setting is not stable
            return -np.inf

    worker_list = []
    for pen_param_res, tree_split in pen_param_results:
        # Need to create model parameters for the full tree since
        # we only trained on a subset of the leaves
        fit_params = pen_param_res.get_fit_params()
        #fit_params.pop('branch_len_inners', None)
        #fit_params.pop('branch_len_offsets_proportion', None)

        # First we need to preserve any bifurcations in the train tree
        for node in tree_split.train_clt.traverse():
            if len(node.get_children()) == 2:
                matching_node = tree_split.val_clt.search_nodes(node_id=node.node_id)[0]
                matching_node.resolved_multifurcation = True

        # Let's start creating the branch lenght assignments for the
        # validation leaves
        spine_lens = pen_param_res.train_history[-1]["spine_lens"]
        dist_to_roots = pen_param_res.train_history[-1]["dist_to_roots"]
        num_tot_nodes = tree_split.val_clt.get_num_nodes()
        num_train_nodes = tree_split.train_clt.get_num_nodes()
        new_br_inners = np.ones(num_tot_nodes) * 1e-10
        new_br_inners[:num_train_nodes] = fit_params['branch_len_inners']
        # We will place the validation leaves at the top of the multifurcation
        # This is a somewhat arbitrary choice.
        # However we definitely cannot maximize validation log lik wrt the validation offsets.
        # Otherwise penalty param picking will not work.
        new_br_offsets = np.ones(num_tot_nodes) * 0.15
        new_br_offsets[:num_train_nodes] = fit_params['branch_len_offsets_proportion']
        for node_id in range(num_train_nodes, num_tot_nodes):
            val_node = tree_split.val_clt.search_nodes(node_id=node_id)[0]
            if not val_node.up.resolved_multifurcation:
                up_id = val_node.up.node_id
                br_inner = fit_params["tot_time"] - dist_to_roots[up_id]
                spine_len = spine_lens[up_id]
                # Place halfway on the spine...
                new_br_offsets[node_id] = spine_len/2/br_inner

        fit_params['branch_len_inners'] = new_br_inners
        fit_params['branch_len_offsets_proportion'] = new_br_offsets
        all_known_params = KnownModelParams(
                target_lams=True,
                tot_time=True,
                indel_params=True)

        transition_wrap_maker = TransitionWrapperMaker(
            tree_split.val_clt,
            tree_split.val_bcode_meta,
            max_extra_steps,
            max_sum_states)
        scorer = LikelihoodScorer(
            get_randint(),  # seed
            tree_split.val_clt,
            tree_split.val_bcode_meta,
            # No iterations because we just want to evaluate a probability
            max_iters=0,
            num_inits=1,
            transition_wrap_maker=transition_wrap_maker,
            fit_param_list=[fit_params],
            known_params=all_known_params,
            scratch_dir=scratch_dir)
        worker_list.append(scorer)

    job_manager = SubprocessManager(
            worker_list,
            None,
            scratch_dir,
            num_processes)
    worker_results = [w[0][0] for w in job_manager.run()]

    # Get Pr(V|T)
    hyperparam_scores = [
            res.log_lik - pen_param_res.log_lik
            for res, (pen_param_res, _) in zip(worker_results, pen_param_results)]
    tot_hyperparam_score = np.mean(hyperparam_scores)
    logging.info("all Pr(Val given T) %s (sum %f)", hyperparam_scores, tot_hyperparam_score)
    return tot_hyperparam_score
