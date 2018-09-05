import numpy as np
from typing import List, Dict

from allele_events import Event
from optim_settings import KnownModelParams
from cell_lineage_tree import CellLineageTree
from likelihood_scorer import LikelihoodScorer
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from split_data import create_kfold_trees, create_kfold_barcode_trees, TreeDataSplit
from tree_distance import TreeDistanceMeasurerAgg
from likelihood_scorer import LikelihoodScorerResult
from parallel_worker import SubprocessManager
import hanging_chad_finder
from common import *

class TuneScorerResult:
    def __init__(self,
            log_barr_pen: float,
            dist_to_half_pen: float,
            model_params_dicts: List[Dict],
            score: float,
            hanging_chad_tuple: Tuple[CellLineageTree, CellLineageTree] = None,
            tree: CellLineageTree = None,
            tree_splits: List[TreeDataSplit] = []):
        """
        @param log_barr_pen: the log barrier penalty param used when fitting the model
        @param dist_to_half_pen: the distance to 0.5 diagonal penalty param used when fitting the model
        @param model_params_dict: an example of the final fitted parameters when we did
                        train/validation split/kfold CV. (this is used mostly as a warm start)
        @param score: assumed to be higher the better
        """
        self.log_barr_pen = log_barr_pen
        self.dist_to_half_pen = dist_to_half_pen
        self.model_params_dicts = model_params_dicts
        self.hanging_chad_tuple = hanging_chad_tuple
        self.score = score
        self.tree = tree
        self.tree_splits = tree_splits

def tune(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args):
    """
    Tunes the `dist_to_half_pen` penalty parameter

    @return List[TuneScorerResult] -- corresponding to each penalty param
                being considered
            As well as the best model params to initialize with
    """
    if args.num_tune_splits <= 0:
        # If no splits, then don't do any tuning
        assert len(args.dist_to_half_pens) == 1
        tune_results = [TuneScorerResult(
            log_barr_pen = args.log_barr,
            dist_to_half_pen = args.dist_to_half_pens[0],
            model_params_dicts = [args.init_params],
            score = 0,
            tree = tree)]
    elif bcode_meta.num_barcodes > 1:
        # For many barcodes, we split by barcode
        tune_results = _tune_hyperparams(
            tree,
            bcode_meta,
            args,
            create_kfold_barcode_trees,
            _get_many_bcode_stability_score)
    else:
        # For single barcode, we split into subtrees
        tune_results = _tune_hyperparams(
            tree,
            bcode_meta,
            args,
            create_kfold_trees,
            _get_one_bcode_stability_score)

        # Cannot use branch lengths as warm start parameters
        # Therefore we remove them from the model param dict
        for res in tune_results:
            for param_dict in res.model_params_dicts:
                param_dict.pop('branch_len_inners', None)
                param_dict.pop('branch_len_offsets_proportion', None)

    pen_param_scores = np.array([res.score for res in tune_results])
    logging.info("Tuning scores %s", pen_param_scores)
    best_idx = np.argmax(pen_param_scores)

    init_model_params = tune_results[best_idx].model_params_dicts[0].copy()
    init_model_params["log_barr_pen"] = args.log_barr
    init_model_params["dist_to_half_pen"] = args.dist_to_half_pens[best_idx]
    return tune_results, init_model_params

def _tune_hyperparams(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        kfold_fnc,
        stability_score_fnc,
        max_num_chad_parents: int = 20):
    """
    @param max_num_chad_parents: max number of chad parents to consider

    @return List[TuneScorerResult] -- corresponding to each hyperparam
                being tuned
    """
    hanging_chad_dict = hanging_chad_finder.get_chads(tree)
    sorted_chad_keys = sorted(list(hanging_chad_dict.keys()))

    # First split the barcode into kfold groups
    tree_splits = kfold_fnc(
            tree,
            bcode_meta,
            args.num_tune_splits)

    all_tune_res = []
    for chad_evt in sorted_chad_keys:
        hanging_chad = hanging_chad_dict[chad_evt]
        print(chad_evt, hanging_chad.node.up.allele_events_list_str)
        chad_tuning_results = []
        print("number of chad parents", len(hanging_chad.possible_parents))
        for chad_par in hanging_chad.possible_parents[:max_num_chad_parents]:
            tree_split_copy = [
                    TreeDataSplit(s.tree.copy(), s.bcode_meta, {})
                    for s in tree_splits]

            # Remove my hanging chad from the orig tree
            tree_copy = tree.copy()
            for node in tree_copy.traverse():
                if node.node_id == hanging_chad.node.node_id:
                    node.detach()
                    break
            # And then add back the hanging chad to the designated parent
            for node in tree_copy.traverse():
                if node.node_id == chad_par.node_id:
                    if node.is_leaf():
                        node.add_child(node.copy())
                    node.add_child(hanging_chad.node.copy())
                    break
            tree_copy.label_node_ids()
            print("attached....", chad_evt, chad_par.allele_events_list_str)
            print(tree_copy.get_ascii(attributes=["allele_events_list_str"]))

            # remove my hanging chad from the kfold splits
            for tree_split in tree_split_copy:
                for node in tree_split.tree.traverse():
                    if node.orig_node_id == hanging_chad.node.node_id:
                        node.detach()
                        break
                #print(tree_split.tree.get_ascii(attributes=["allele_events_list_str"]))

            # Now add my new possibility
            for tree_split in tree_split_copy:
                for node in tree_split.tree.traverse():
                    if node.orig_node_id == chad_par.node_id:
                        node.add_child(hanging_chad.node.copy())
                        break
                tree_split.tree.label_node_ids()
                #print(tree_split.tree.get_ascii(attributes=["allele_events_list_str"]))
                #print("attached", chad_evt, chad_par.allele_events_list_str)

            tuning_results = _get_tuning_results(
                tree_copy,
                tree_split_copy,
                args,
                stability_score_fnc,
                chad_tuple = (hanging_chad.node, chad_par))
            chad_tuning_results.append(tuning_results)

        best_hanging_chad_idx = np.argmax([
                np.max([r.score for r in tune_res]) for tune_res in chad_tuning_results])
        for chad_par, tune_res in zip(hanging_chad.possible_parents, chad_tuning_results):
            print(chad_par.allele_events_list_str, [r.score for r in tune_res])
        all_tune_res += chad_tuning_results

        #TODO: remove this. we are only trying one hanging chad first
        break
    print(all_tune_res)
    # TODO: right now we just flatten the list
    return [r for res in all_tune_res for r in res]

def _get_tuning_results(
        tree: CellLineageTree,
        tree_splits: List[TreeDataSplit],
        args,
        stability_score_fnc,
        chad_tuple = None):
    """
    @return List[TuneScorerResult]
    """
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
        init_model_param_list = init_model_param_list,
        known_params = args.known_params,
        abundance_weight = args.abundance_weight)
        for tree_split, transition_wrap_maker in zip(tree_splits, trans_wrap_makers)]
    job_manager = SubprocessManager(
            worker_list,
            None,
            args.scratch_dir,
            threads=args.num_processes)
    train_results = [r for r, _ in job_manager.run()]

    # Now find the best penalty param by finding the most stable one
    # Stability is defined as the least variable target lambda estimates and branch length estimates
    tune_results = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        res_folds = [train_res[idx] for train_res in train_results]
        stability_score = stability_score_fnc(res_folds, tree_splits, tree)

        # Create our summary of tuning
        tune_result = TuneScorerResult(
            args.log_barr,
            dist_to_half_pen,
            [res.model_params_dict for res in res_folds if res is not None],
            stability_score,
            hanging_chad_tuple = chad_tuple,
            tree = tree,
            tree_splits = tree_splits)
        for init_model_params in tune_result.model_params_dicts:
            init_model_params["log_barr_pen"] = args.log_barr
            init_model_params["dist_to_half_pen"] = args.dist_to_half_pens[idx]

        tune_results.append(tune_result)
        logging.info(
                "Pen param %f stability score %s",
                dist_to_half_pen,
                tune_result.score)

    return tune_results

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
        orig_tree: CellLineageTree,
        weight: float = 0):
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

            # Also retrieve branch length estimates
            # Just the leaf branches...
            #final_branch_lens = pen_param_res.train_history[-1]["branch_lens"]
            #for new_id, orig_id in tree_split.node_to_orig_id.items():
            #    tree_param_ests[orig_id].append(final_branch_lens[new_id])

            ## now the internal branches...
            #leaf_id_dict = {}
            #for leaf in pen_param_res.fitted_bifurc_tree:
            #    leaf_id_dict[tree_split.node_to_orig_id[leaf.node_id]] = leaf
            #fitted_leaf_ids = set(list(leaf_id_dict.keys()))

            #for node in orig_tree.traverse('postorder'):
            #    if not node.is_leaf():
            #        print(node.leaf_ids)
            #        if set(node.leaf_ids).issubset(fitted_leaf_ids):
            #            tree_key = tuple(node.leaf_ids)
            #            leaf_nodes = [leaf_id_dict[leaf_id] for leaf_id in node.leaf_ids]
            #            mrca = leaf_nodes[0].get_common_ancestor(*(leaf_nodes[1:]))
            #            fitted_dist = pen_param_res.fitted_bifurc_tree.get_distance(mrca)
            #            if tree_key in tree_param_ests:
            #                tree_param_ests[tree_key].append(fitted_dist)
            #            else:
            #                tree_param_ests[tree_key] = [fitted_dist]

            ## Also add the branch going to the root
            #tree_param_ests[0].append(
            #        pen_param_res.fitted_bifurc_tree.get_children()[0].dist)
        else:
            logging.info("had trouble training. very instable")
            is_stable = False
            break

    if is_stable:
        mean_target_param_est = sum(target_param_ests)/len(target_param_ests)
        targ_stability_score = -np.mean([
            np.power(np.linalg.norm(targ_param_est - mean_target_param_est), 2)
            for targ_param_est in target_param_ests])/np.power(np.linalg.norm(mean_target_param_est), 2)

        #for leaf_id in tree_param_ests.keys():
        #    tree_param_ests[leaf_id] = np.array(tree_param_ests[leaf_id])

        #tree_stability_score = -np.mean([
        #    np.mean(np.power(len_ests - np.mean(len_ests), 2))/np.power(np.mean(len_ests), 2)
        #    for len_ests in tree_param_ests.values() if len(len_ests) > 1])
        tree_stability_score = 0

        stability_score = weight * tree_stability_score + (1 - weight) * targ_stability_score

        logging.info("mean targ %s", mean_target_param_est)
        logging.info("stability scores %s %s", targ_stability_score, tree_stability_score)
        return stability_score
    else:
        return -np.inf
