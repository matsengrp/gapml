import numpy as np
from typing import List, Dict

from optim_settings import KnownModelParams
from cell_lineage_tree import CellLineageTree
from likelihood_scorer import LikelihoodScorer
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from split_data import create_kfold_trees
from split_data import create_kfold_barcode_trees
from tree_distance import TreeDistanceMeasurerAgg
from common import *

class TuneScorerResult:
    def __init__(self,
            log_barr_pen: float,
            dist_to_half_pen: float,
            model_params_dict: Dict,
            score: float):
        """
        @param score: assumed to be higher the better
        """
        self.log_barr_pen = log_barr_pen
        self.dist_to_half_pen = dist_to_half_pen
        self.model_params_dict = model_params_dict
        self.score = score

def tune(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    Tunes the `dist_to_half_pen` penalty parameter
    @return List[List[TuneScorerResult]]
    """
    best_params = args.init_params
    best_params["log_barr_pen"] = args.log_barr
    best_params["dist_to_half_pen"] = args.dist_to_half_pens[0]
    if len(args.dist_to_half_pens) == 1:
        # There is nothing to tune -- no fitting to do.
        # Sends back the same initialization params.
        return [[TuneScorerResult(
            args.log_barr,
            args.dist_to_half_pens,
            best_params,
            score = 0)]]

    if bcode_meta.num_barcodes > 1:
        validation_res = _tune_hyperparams_one_split_many_bcodes(
            tree,
            bcode_meta,
            args,
            oracle_dist_measurers)
        validation_results = [validation_res]
    else:
        validation_res = _tune_hyperparams_single_bcode(
            tree,
            bcode_meta,
            args)
        validation_results = [validation_res]
    return validation_results

def _tune_hyperparams_one_split_many_bcodes(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    Hyperparams are tuned by splitting independent barcodes.
    The score is the log likelihood of the validation barcodes.

    @return List[TuneScorerResult]
    """
    assert bcode_meta.num_barcodes > 1
    # First split the data into training vs validation
    tree_splits = create_kfold_barcode_trees(
            tree,
            bcode_meta,
            args.num_tune_splits)

    trans_wrap_makers = [TransitionWrapperMaker(
            tree_split.tree,
            tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states) for tree_split in tree_splits]

    # First create the initialization/optimization settings
    init_model_param_list = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        if idx == 0:
            new_init_model_params = args.init_params.copy()
        else:
            new_init_model_params = {}
        new_init_model_params["log_barr_pen"] = args.log_barr
        new_init_model_params["dist_to_half_pen"] = dist_to_half_pen
        init_model_param_list.append(new_init_model_params)

    # Actually fit the subsampled trees
    train_results = [LikelihoodScorer(
        get_randint(),
        tree_split.tree,
        tree_split.bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        init_model_param_list = init_model_param_list,
        known_params = args.known_params,
        abundance_weight = args.abundance_weight).run_worker(None)
        for tree_split, transition_wrap_maker in zip(tree_splits, trans_wrap_makers)]

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates
    final_validation_results = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        targ_param_ests = []
        tree_param_ests = []
        is_stable = True
        for train_res in train_results:
            pen_param_res = train_res[idx]
            if pen_param_res is not None:
                param_dict = pen_param_res.model_params_dict
                targ_param_ests.append(np.concatenate([
                    param_dict["target_lams"],
                    param_dict["double_cut_weight"],
                    param_dict["trim_long_factor"]]))
                tree_param_ests.append(np.concatenate([
                    params_dict['branch_len_inners'],
                    params_dict['branch_len_offsets_proportion']]))
            else:
                logging.info("had trouble training. very instable")
                is_stable = False
                break
        if is_stable:
            mean_targ_param_est = sum(targ_param_ests)/len(targ_param_ests)
            mean_tree_param_est = sum(tree_param_ests)/len(tree_param_ests)
            logging.info("all params... %s %s", mean_targ_param_est, mean_tree_param_est)
            tree_stability_score = -np.mean([
                np.power(np.linalg.norm(tree_param_est - mean_tree_param_est), 2)
                for tree_param_est in tree_param_ests])/np.power(np.linalg.norm(mean_tree_param_est), 2)
            targ_stability_score = -np.mean([
                np.power(np.linalg.norm(targ_param_est - mean_targ_param_est), 2)
                for targ_param_est in targ_param_ests])/np.power(np.linalg.norm(mean_targ_param_est), 2)
            stability_score = tree_stability_score + targ_stability_score

            # Create our summary of tuning
            tune_result = TuneScorerResult(
                args.log_barr,
                dist_to_half_pen,
                # Copy over model parameter estimates from one of the trained subsampled trees
                # Doesn't matter which one.
                pen_param_res.model_params_dict,
                stability_score)
            final_validation_results.append(tune_result)
            logging.info(
                    "Pen param %f stability score %f",
                    dist_to_half_pen,
                    tune_result.score)
        else:
            # Create our summary of tuning -- not stable
            tune_result = TuneScorerResult(
                args.log_barr,
                dist_to_half_pen,
                None,
                -np.inf)
            final_validation_results.append(tune_result)
            logging.info("Pen param %f NOT STABLE", dist_to_half_pen)

    return final_validation_results

def _tune_hyperparams_single_bcode(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args):
    """
    Tunes the hyperparams by splitting trees into kfold subtrees
    and fitting different hyperparam. Scores each hyperparam by the negative
    of the variance of the target lambda estimate
    @return List[TuneScorerResult]
    """
    assert bcode_meta.num_barcodes == 1
    # First create subsampled kfold trees
    tree_splits = create_kfold_trees(
            tree,
            bcode_meta,
            args.num_tune_splits)

    trans_wrap_makers = [TransitionWrapperMaker(
            tree_split.tree,
            tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states) for tree_split in tree_splits]

    # First create the initialization/optimization settings
    init_model_param_list = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        if idx == 0:
            new_init_model_params = args.init_params.copy()
        else:
            new_init_model_params = {}
        new_init_model_params["log_barr_pen"] = args.log_barr
        new_init_model_params["dist_to_half_pen"] = dist_to_half_pen
        init_model_param_list.append(new_init_model_params)

    # Actually fit the subsampled trees
    train_results = [LikelihoodScorer(
        get_randint(),
        tree_split.tree,
        tree_split.bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        init_model_param_list = init_model_param_list,
        known_params = args.known_params,
        abundance_weight = args.abundance_weight).run_worker(None)
        for tree_split, transition_wrap_maker in zip(tree_splits, trans_wrap_makers)]

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates
    final_validation_results = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        target_param_ests = []
        is_stable = True
        for train_res in train_results:
            pen_param_res = train_res[idx]
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
            logging.info("mean targ %s", mean_target_param_est)
            stability_score = -np.mean([
                np.power(np.linalg.norm(targ_param_est - mean_target_param_est), 2)
                for targ_param_est in target_param_ests])

            # Copy over model parameter estimates from one of the trained subsampled trees
            # Doesn't matter which one.
            example_result = pen_param_res
            example_result.model_params_dict.pop('branch_len_inners', None)
            example_result.model_params_dict.pop('branch_len_offsets_proportion', None)

            # Create our summary of tuning
            tune_result = TuneScorerResult(
                args.log_barr,
                dist_to_half_pen,
                example_result.model_params_dict,
                stability_score)
            final_validation_results.append(tune_result)
            logging.info(
                    "Pen param %f stability score %f",
                    dist_to_half_pen,
                    tune_result.score)
        else:
            # Create our summary of tuning -- not stable
            tune_result = TuneScorerResult(
                args.log_barr,
                dist_to_half_pen,
                None,
                -np.inf)
            final_validation_results.append(tune_result)
            logging.info("Pen param %f NOT STABLE", dist_to_half_pen)

    return final_validation_results
