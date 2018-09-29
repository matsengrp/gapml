import numpy as np
from typing import List, Dict
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
from clt_likelihood_penalization import mark_target_status_to_penalize
import ancestral_events_finder as anc_evt_finder
from optim_settings import KnownModelParams


class PenaltyScorerResult:
    def __init__(
            self,
            score: float,
            fit_results: List[LikelihoodScorerResult]):
        """
        @param log_barr_pen_param: the log barrier penalty param used when fitting the model
        @param dist_to_half_pen_param: the distance to 0.5 diagonal penalty param used when fitting the model
        @param model_params_dict: an example of the final fitted parameters when we did
                        train/validation split/kfold CV. (this is used mostly as a warm start)
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
        assessor: ModelAssessor):
    """
    Tunes the `dist_to_half_pen_param`, `target_lam_pen_param` penalty parameters

    @return PenaltyTuneResult
    """
    assert len(args.dist_to_half_pen_params) > 1 or len(args.target_lam_pen_params) > 1

    if bcode_meta.num_barcodes > 1:
        # For many barcodes, we split by barcode
        return _tune_hyperparams(
            tree,
            bcode_meta,
            args,
            fit_params,
            create_kfold_barcode_trees,
            _get_many_bcode_stability_score,
            assessor)
    else:
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        # For single barcode, we split into subtrees
        return _tune_hyperparams(
            tree,
            bcode_meta,
            args,
            fit_params,
            create_kfold_trees,
            _get_one_bcode_stability_score,
            assessor)


def _tune_hyperparams(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        kfold_fnc,
        stability_score_fnc,
        assessor: ModelAssessor = None,
        conv_thres: float = 1e-5):
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
    logging.info("hyperparam tuning %d splits", n_splits)
    all_tree_splits = kfold_fnc(
            tree,
            bcode_meta,
            n_splits)
    random.shuffle(all_tree_splits)
    tree_splits = all_tree_splits[:args.max_fit_splits]

    trans_wrap_makers = [TransitionWrapperMaker(
            tree_split.tree,
            tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states) for tree_split in tree_splits]
    # Mark the target status to penalize for each node in the tree
    for tree_split in tree_splits:
        anc_evt_finder.annotate_ancestral_states(tree_split.tree, bcode_meta)
        mark_target_status_to_penalize(tree_split.tree)

    # First create the initialization/optimization settings
    fit_param_list = []
    for dist_to_half_pen_param in args.dist_to_half_pen_params:
        for target_lam_pen_param in args.target_lam_pen_params:
            if len(fit_param_list) == 0:
                new_fit_params = fit_params.copy()
            else:
                new_fit_params = {}
            new_fit_params["log_barr_pen_param"] = args.log_barr_pen_param
            new_fit_params["dist_to_half_pen_param"] = dist_to_half_pen_param
            new_fit_params["target_lam_pen_param"] = target_lam_pen_param
            new_fit_params["conv_thres"] = conv_thres
            fit_param_list.append(new_fit_params)

    # Actually fit the trees using the kfold barcodes
    # TODO: if one of the penalty params fails, then remove it from the subsequent
    # kfold runs
    worker_list = [LikelihoodScorer(
        get_randint(),
        tree_split.tree,
        tree_split.bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        fit_param_list=fit_param_list,
        known_params=args.known_params,
        scratch_dir=args.scratch_dir,
        assessor=assessor,
        assess_bcode_idxs=tree_split.train_bcode_idxs)
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

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates and branch length estimates
    tune_results = []
    for idx, fit_param in enumerate(fit_param_list):
        dist_to_half_pen_param = fit_param['dist_to_half_pen_param']
        target_lam_pen_param = fit_param['target_lam_pen_param']
        logging.info(
                "PEN PARAM setting %d: dist_to_half_pen_param %f target_lam_pen_param %f",
                idx,
                fit_param['dist_to_half_pen_param'],
                fit_param['target_lam_pen_param'])
        res_folds = [train_res[idx] for train_res in train_results]
        stability_score = stability_score_fnc(res_folds, tree_splits, tree)

        # Create our summary of tuning
        tune_result = PenaltyScorerResult(
            stability_score,
            res_folds)
        tune_results.append(tune_result)
        logging.info(
                "Pen param dist_to_half %f, target_lam %f, stability score %s",
                dist_to_half_pen_param,
                target_lam_pen_param,
                tune_result.score)

    return PenaltyTuneResult(
                tree,
                tree_splits,
                tune_results)


def _get_many_bcode_stability_score(
        pen_param_results: List[LikelihoodScorerResult],
        tree_splits: List[TreeDataSplit],
        orig_tree: CellLineageTree,
        max_num_leaves: int = None):
    """
    """
    is_stable = True
    for pen_param_res in pen_param_results:
        if pen_param_res is None:
            is_stable = False
            break

    if is_stable:
        worker_list = []
        for pen_param_res, tree_split in zip(pen_param_results, tree_splits):
            fit_params = pen_param_res.get_fit_params()
            all_known_params = KnownModelParams(
                target_lams=True,
                tot_time=True,
                indel_params=True)
            anc_evt_finder.annotate_ancestral_states(tree_split.val_clt, tree_split.bcode_meta)
            mark_target_status_to_penalize(tree_split.val_clt)
            transition_wrap_maker = TransitionWrapperMaker(
                tree_split.val_clt,
                tree_split.val_bcode_meta,
                100,
                1000)
            scorer = LikelihoodScorer(
                1, # seed
                tree_split.val_clt,
                tree_split.val_bcode_meta,
                max_iters=0, #50000,
                num_inits=1,
                transition_wrap_maker=transition_wrap_maker,
                fit_param_list=[fit_params],
                known_params=all_known_params,
                scratch_dir="_output/scratch")
            worker_list.append(scorer)

        job_manager = SubprocessManager(
                worker_list,
                None,
                "_output/scratch",
                10)
        worker_results = [w[0][0] for w in job_manager.run()]
        val_log_liks = [res.log_lik for res in worker_results]

        val_log_lik = np.sum(val_log_liks)
        logging.info("all hyperparam split-scores %s", val_log_liks)
        logging.info("hyperparam sum-split-scores %s", val_log_lik)
        return val_log_lik
    else:
        return -np.inf

def _get_many_bcode_stability_score_old(
        pen_param_results: List[LikelihoodScorerResult],
        weight: float = 0.5):
    """
    The stability score for kfold barcode trees is a convex combination of the
    normalized variance of the target lambdas as well as the normalized variance
    of the branch length parameters

    @return the stability score (float) -- if training failed for any replicates,
            the stability score is -inf
    """
    assert weight >= 0
    is_stable = True
    targ_param_ests = []
    tree_param_ests = []
    for pen_param_res in pen_param_results:
        if pen_param_res is not None:
            param_dict = pen_param_res.model_params_dict
            targ_param_ests.append(np.concatenate([
                param_dict["target_lams"],
                param_dict["double_cut_weight"],
                param_dict["trim_long_factor"]]))
            tree_param_ests.append(np.concatenate([
                param_dict['branch_len_inners'],
                param_dict['branch_len_offsets_proportion']]))
        else:
            logging.info("had trouble training. very unstable")
            is_stable = False
            break

    if is_stable:
        mean_targ_param_est = sum(targ_param_ests)/len(targ_param_ests)
        mean_tree_param_est = sum(tree_param_ests)/len(tree_param_ests)
        tree_stability_score = -np.mean([
            np.power(np.linalg.norm(tree_param_est - mean_tree_param_est), 2)
            for tree_param_est in tree_param_ests])
        targ_stability_score = -np.mean([
            np.power(np.linalg.norm(targ_param_est - mean_targ_param_est), 2)
            for targ_param_est in targ_param_ests])
        stability_score = weight * tree_stability_score + (1 - weight) * targ_stability_score

        logging.info("all params... %s %s", mean_targ_param_est, mean_tree_param_est)
        logging.info("stability scores %s %s", targ_stability_score, tree_stability_score)
        logging.info("tree var %s", np.mean([
            np.power(np.linalg.norm(tree_param_est - mean_tree_param_est), 2)
            for tree_param_est in tree_param_ests]))
        logging.info("tree po %s", np.power(np.linalg.norm(mean_tree_param_est), 2))
        logging.info("targ var %s", np.mean([
            np.power(np.linalg.norm(targ_param_est - mean_targ_param_est), 2)
            for targ_param_est in targ_param_ests]))
        logging.info("targ po %s", np.power(np.linalg.norm(mean_targ_param_est), 2))

        return stability_score
    else:
        return -np.inf


def _get_one_bcode_stability_score(
        pen_param_results: List[LikelihoodScorerResult],
        tree_splits: List[TreeDataSplit],
        orig_tree: CellLineageTree,
        max_num_leaves: int = None):
    """
    @param weight: weight on the tree stability score (0 to 1)
    @return stability score -- is the variance of the fitted target lambdas across folds
    """
    target_param_ests = []
    is_stable = True
    scratch_dir = "_output/scratch"
    targ_stability_scores = []
    worker_list = []
    for pen_param_res, tree_split in zip(pen_param_results, tree_splits):
        if pen_param_res is None:
            logging.info("had trouble training. very instable")
            is_stable = False
            break

        param_dict = pen_param_res.model_params_dict
        target_param_ests.append(np.concatenate([
            param_dict["target_lams"],
            param_dict["double_cut_weight"],
            param_dict["trim_long_factor"]]))

        num_nodes = tree_split.tree.get_num_nodes()
        num_val_nodes = len(tree_split.val_obs)
        fit_params = pen_param_res.get_fit_params()
        new_br_inners_mask = np.zeros(num_nodes + num_val_nodes, dtype=bool)
        new_br_inners_mask[:num_nodes] = True
        new_br_offsets_mask = np.zeros(num_nodes + num_val_nodes, dtype=bool)
        new_br_offsets_mask[:num_nodes] = True
        new_br_inners = np.ones(num_nodes + num_val_nodes) * 1e-10
        new_br_inners[:num_nodes] = fit_params['branch_len_inners']
        new_br_offsets = np.ones(num_nodes + num_val_nodes) * 0.1
        new_br_offsets[:num_nodes] = fit_params['branch_len_offsets_proportion']
        train_val_tree = tree_split.tree.copy()
        for node in train_val_tree.traverse():
            if len(node.get_children()) == 2:
                node.resolved_multifurcation = True
        for leaf_idx, (orig_leaf, leaf_par_id) in enumerate(tree_split.val_obs):
            leaf = orig_leaf.copy()
            leaf.add_feature("node_id", leaf_idx + num_nodes)
            parent_node = train_val_tree.search_nodes(orig_node_id=leaf_par_id)[0]
            parent_node.add_child(leaf)
        all_known_params = KnownModelParams(
                target_lams=True,
                tot_time=True,
                indel_params=True,
                branch_lens=True,
                branch_len_inners=new_br_inners_mask,
                branch_len_offsets_proportion=new_br_offsets_mask)
        fit_params['branch_len_inners'] = new_br_inners
        fit_params['branch_len_offsets_proportion'] = new_br_offsets
        fit_params['dist_to_half_pen_param'] = 0

        anc_evt_finder.annotate_ancestral_states(train_val_tree, tree_split.bcode_meta)
        mark_target_status_to_penalize(train_val_tree)
        transition_wrap_maker = TransitionWrapperMaker(
            train_val_tree,
            tree_split.bcode_meta,
            100,
            1000)
        scorer = LikelihoodScorer(
            1, # seed
            train_val_tree,
            tree_split.bcode_meta,
            max_iters=0, #50000,
            num_inits=1,
            transition_wrap_maker=transition_wrap_maker,
            fit_param_list=[fit_params],
            known_params=all_known_params,
            scratch_dir=scratch_dir)
        worker_list.append(scorer)

    job_manager = SubprocessManager(
            worker_list,
            None,
            "_output/scratch",
            10)
    worker_results = [w[0][0] for w in job_manager.run()]
    #targ_stability_scores = [
    #        res.log_lik/len(tree_split.val_obs)
    #        for res, tree_split in zip(
    #            worker_results,
    #            tree_splits)]
    targ_stability_scores = [
            res.log_lik - pen_param_res.log_lik
            for res, pen_param_res in zip(
                worker_results,
                pen_param_results)]

    targ_stability_score = np.sum(targ_stability_scores)
    logging.info("all stability scores %s", targ_stability_scores)

    if is_stable:
        mean_target_param_est = sum(target_param_ests)/len(target_param_ests)
        logging.info("mean targ %s", mean_target_param_est)
        logging.info("stability scores %s", targ_stability_score)
        stability_score = targ_stability_score
        return stability_score
    else:
        return -np.inf
