import numpy as np
from typing import List, Dict
import logging

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from split_data import create_kfold_trees, create_kfold_barcode_trees, TreeDataSplit
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from parallel_worker import SubprocessManager
from common import get_randint
from model_assessor import ModelAssessor


class PenaltyScorerResult:
    def __init__(
            self,
            score: float,
            log_barr_pen_param: float,
            dist_to_half_pen_param: float,
            fit_results: List[LikelihoodScorerResult]):
        """
        @param log_barr_pen_param: the log barrier penalty param used when fitting the model
        @param dist_to_half_pen_param: the distance to 0.5 diagonal penalty param used when fitting the model
        @param model_params_dict: an example of the final fitted parameters when we did
                        train/validation split/kfold CV. (this is used mostly as a warm start)
        @param score: assumed to be higher the better
        """
        self.score = score
        self.log_barr_pen_param = log_barr_pen_param
        self.dist_to_half_pen_param = dist_to_half_pen_param
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

        # TODO: how do we warm start using previous branch length estimates?
        fit_params = chosen_best_res.get_fit_params()
        # Popping branch length estimates right now because i dont know
        # how to warm start using these estimates...?
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        return fit_params, chosen_best_res


def tune(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor):
    """
    Tunes the `dist_to_half_pen_param` penalty parameter

    @return PenaltyTuneResult
    """
    assert len(args.dist_to_half_pen_params) > 1

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
        assessor: ModelAssessor = None):
    """
    @param max_num_chad_parents: max number of chad parents to consider

    @return List[PenaltyScorerResult] -- corresponding to each hyperparam
                being tuned
    """
    # First split the barcode into kfold groups
    tree_splits = kfold_fnc(
            tree,
            bcode_meta,
            args.num_penalty_tune_splits)

    trans_wrap_makers = [TransitionWrapperMaker(
            tree_split.tree,
            tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states) for tree_split in tree_splits]

    # First create the initialization/optimization settings
    fit_param_list = []
    for idx, dist_to_half_pen_param in enumerate(args.dist_to_half_pen_params):
        if idx == 0:
            new_fit_params = fit_params.copy()
        else:
            new_fit_params = {}
        new_fit_params["log_barr_pen_param"] = args.log_barr_pen_param
        new_fit_params["dist_to_half_pen_param"] = dist_to_half_pen_param
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

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates and branch length estimates
    tune_results = []
    for idx, dist_to_half_pen_param in enumerate(args.dist_to_half_pen_params):
        res_folds = [train_res[idx] for train_res in train_results]
        stability_score = stability_score_fnc(res_folds, tree_splits, tree)

        # Create our summary of tuning
        tune_result = PenaltyScorerResult(
            stability_score,
            args.log_barr_pen_param,
            dist_to_half_pen_param,
            res_folds)
        tune_results.append(tune_result)
        logging.info(
                "Pen param %f stability score %s",
                dist_to_half_pen_param,
                tune_result.score)

    return PenaltyTuneResult(
                tree,
                tree_splits,
                tune_results)


def _get_many_bcode_stability_score(
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
            for tree_param_est in tree_param_ests])/np.power(np.linalg.norm(mean_tree_param_est), 2)
        targ_stability_score = -np.mean([
            np.power(np.linalg.norm(targ_param_est - mean_targ_param_est), 2)
            for targ_param_est in targ_param_ests])/np.power(np.linalg.norm(mean_targ_param_est), 2)
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
        orig_tree: CellLineageTree):
    """
    @param weight: weight on the tree stability score (0 to 1)
    @return stability score -- is the variance of the fitted target lambdas across folds
    """
    for node in orig_tree.traverse('postorder'):
        if node.is_leaf():
            node.add_feature("leaf_ids", [node.node_id])
        else:
            node.add_feature("leaf_ids", [
                leaf_id for c in node.children for leaf_id in c.leaf_ids])

    target_param_ests = []
    tree_param_ests = {leaf.node_id: [] for leaf in orig_tree}
    tree_param_ests[0] = []
    is_stable = True
    for pen_param_res, tree_split in zip(pen_param_results, tree_splits):
        if pen_param_res is not None:
            param_dict = pen_param_res.model_params_dict
            target_param_ests.append(np.concatenate([
                param_dict["target_lams"],
                param_dict["double_cut_weight"],
                param_dict["trim_long_factor"]]))
        else:
            logging.info("had trouble training. very instable")
            is_stable = False
            break

    if is_stable:
        mean_target_param_est = sum(target_param_ests)/len(target_param_ests)
        targ_stability_score = -np.mean([
            np.power(np.linalg.norm(targ_param_est - mean_target_param_est), 2)
            for targ_param_est in target_param_ests])/np.power(np.linalg.norm(mean_target_param_est), 2)

        stability_score = targ_stability_score

        logging.info("mean targ %s", mean_target_param_est)
        logging.info("stability scores %s", targ_stability_score)
        return stability_score
    else:
        return -np.inf
