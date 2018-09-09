import numpy as np
from typing import List, Dict
import logging
import scipy.stats

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import SubprocessManager
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from common import get_randint
from model_assessor import ModelAssessor
from tree_distance import * # BHVDistanceMeasurer, InternalCorrMeasurer, InternalHeightsMeasurer, TreeDistanceMeasurer
from optim_settings import KnownModelParams


"""
Hanging chad is our affectionate name for the inter-target cuts that have ambiguous placement
in the tree. For example, if we don't know if the 1-3 inter-target cut is a child of the root
node or a child of another allele with a 2-2 intra-target cut.
"""


class HangingChadResult:
    def __init__(
            self,
            targ_score: float,
            tree_score: float,
            chad_node: CellLineageTree,
            parent_node: CellLineageTree,
            fit_res: LikelihoodScorerResult,
            weight: float):
        self.targ_score = targ_score
        self.tree_score = tree_score
        self.chad_node = chad_node
        self.parent_node = parent_node
        self.fit_res = fit_res
        self.weight = weight

    @property
    def score(self):
        return self.weight * self.targ_score + (1 - self.weight) * self.tree_score

    def __str__(self):
        return "%s=>%s, score=%f (targ=%f, tree=%f)" % (
                self.parent_node.allele_events_list_str,
                self.chad_node.allele_events_list_str,
                self.score,
                self.targ_score,
                self.tree_score)


class HangingChadTuneResult:
    """
    Stores the results when we considered all the different possible places
    to place our hanging chad in the tree. Includes the fitted results when we
    removed the hanging chad from the tree entirely and the results for each possible
    place in the tree.
    """
    def __init__(
            self,
            no_chad_res: LikelihoodScorerResult,
            new_chad_results: List[HangingChadResult]):
        """
        @param no_chad_res: result when we fit the tree without the hanging chad
        @param new_chad_results: each result when we place the hanging chad in the different
                            possible locations in the tree
        """
        self.no_chad_res = no_chad_res
        self.new_chad_results = new_chad_results

    def get_best_result(self):
        best_chad_idx = np.argmax([
                chad_res.score for chad_res in self.new_chad_results])
        best_chad = self.new_chad_results[best_chad_idx]
        best_fit_res = best_chad.fit_res

        # TODO: how do we warm start using previous branch length estimates?
        fit_params = best_fit_res.get_fit_params()
        # Popping branch length estimates right now because i dont know
        # how to warm start using these estimates...?
        # fit_params.pop('branch_len_inners', None)
        # fit_params.pop('branch_len_offsets_proportion', None)
        return best_fit_res.orig_tree, fit_params, best_fit_res


class HangingChad:
    def __init__(
            self,
            node: CellLineageTree,
            possible_parents: List[CellLineageTree],
            parsimony_contribution: int):
        """
        @param node: the hanging chad
        @param possible_parents: possible nodes that can be the parent of this node
                                while preserving the total parsimony score of the tree
        @param parsimony_contribution: the contributio to the total parsimony score
                                from this hanging chad
        """
        self.node = node
        self.parsimony_contribution = parsimony_contribution
        self.possible_parents = possible_parents

    def __str__(self):
        return "%s<=%s" % (
            self.node.allele_events_list_str,
            [p.allele_events_list_str for p in self.possible_parents])


def _get_parsimony_score(node: CellLineageTree, up_node: CellLineageTree):
    """
    Gets the parsimony score on a single potential branch
    @param node: a potential child node
    @param up_node: a potential parent node
    @return parsimony score, returns None if this branch is impossible
    """
    node_evts = [set(allele_evts.events) for allele_evts in node.allele_events_list]
    node_up_evts = [set(allele_evts.events) for allele_evts in up_node.allele_events_list]

    # Check if all events in the parent node are hidden or are in the child node
    hides_all_remain = all([
        any([node_evt.hides(remain_node_up_evt) for node_evt in n_evts])
        for n_evts, n_up_evts in zip(node_evts, node_up_evts)
        for remain_node_up_evt in n_up_evts - n_evts])

    if not hides_all_remain:
        return None

    # Get the actual parsimony score now
    num_evts = sum([
        len(n_evts - n_up_evts) for n_evts, n_up_evts in zip(node_evts, node_up_evts)])
    return num_evts


def get_chads(tree: CellLineageTree):
    """
    Find the hanging chads in the tree without increasing the parsimony score

    @param tree: the tree that we are supposed to find hanging chads in
    @return List[HangingChad]
    """
    hanging_chads = []
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue

        # Get node's parsimony score
        parsimony_score = _get_parsimony_score(node, node.up)
        assert parsimony_score is not None

        # Determine if there is another place in the tree where we can preserve
        # parsimony score
        # TODO: this still does not find every possible location for the hanging chads
        # it is possible that there is a node omitted in the parsimony tree that
        # this node can be a child of
        # Note: we are using a dictionary and preorder traversal so we don't consider
        # hanging the chad off of two nodes with the same allele_events_list_str.
        # In particular, we want to hang off of the earliest occurance of the allele_events_list_str
        # since that is the multifurc location. The other node with the same allele_events_list_str
        # is going to also dangle off the same multifurc so it is pointless to consider
        # hanging the chad off of this other node. Instead we handle that case by the continuous
        # topology tuner
        possible_parents = {}
        for potential_par_node in tree.traverse("preorder"):
            if potential_par_node.allele_events_list_str in possible_parents:
                continue

            potential_score = _get_parsimony_score(node, potential_par_node)
            if potential_score is None or parsimony_score == 0:
                continue

            if potential_score == parsimony_score:
                possible_parents[potential_par_node.allele_events_list_str] = potential_par_node

        if len(possible_parents) > 1:
            hanging_chads.append(HangingChad(
                    node,
                    list(possible_parents.values()),
                    parsimony_score))

    logging.info("Number of hanging chads found: %d", len(hanging_chads))
    return hanging_chads

def get_init_target_lams(target_lams_old, mean_val=0):
    random_perturb = np.random.uniform(size=target_lams_old.size) * 0.001
    random_perturb = random_perturb - np.mean(random_perturb)
    random_perturb[0] = 0
    return np.exp(mean_val * np.ones(target_lams_old.size) + random_perturb)

def _prepare_nochad_fit_params(
        nochad_tree: CellLineageTree,
        fit_params: Dict,
        num_nochad_nodes: int,
        known_params: KnownModelParams):
    if 'branch_len_inners' not in fit_params:
        return
    if not known_params.target_lams:
        fit_params['target_lams'] = get_init_target_lams(fit_params['target_lams'])
    fit_params.pop('branch_len_inners', None)
    fit_params.pop('branch_len_offsets_proportion', None)
    #prev_branch_inners = fit_params['branch_len_inners']
    #prev_branch_offsets_proportion = fit_params['branch_len_offsets_proportion']
    #fit_params['branch_len_inners'] = np.ones(num_nochad_nodes) * 1e-10
    #fit_params['branch_len_offsets_proportion'] = np.random.rand(num_nochad_nodes) * 0.5
    #for node in nochad_tree.traverse():
    #    fit_params['branch_len_inners'][node.node_id] = prev_branch_inners[node.orig_node_id]
    #    fit_params['branch_len_offsets_proportion'][node.node_id] = prev_branch_offsets_proportion[node.orig_node_id]


def tune(
        hanging_chad: HangingChad,
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor = None):
    """
    Tune the given hanging chad
    @return HangingChadTuneResult
    """
    assert len(hanging_chad.possible_parents) > 1

    # Remove my hanging chad from the orig tree
    # Also track the parent we took it off from so that
    # we always have an option of NOT moving the hanging chad
    nochad_tree = tree.copy()
    current_parent_id = None
    for node in nochad_tree.traverse("preorder"):
        node.add_feature("orig_node_id", node.node_id)
        if node.node_id == hanging_chad.node.node_id:
            current_parent_id = node.up.orig_node_id
            logging.info("current parent of chad %s", node.up.allele_events_list_str)
            node.detach()
    assert current_parent_id is not None
    num_nochad_nodes = nochad_tree.label_node_ids()
    logging.info("no chad tree leaves %d", len(nochad_tree))

    _prepare_nochad_fit_params(nochad_tree, fit_params, num_nochad_nodes, args.known_params)

    # Now fit the tree without the hanging chad
    trans_wrap_maker = TransitionWrapperMaker(
            nochad_tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    no_chad_res = LikelihoodScorer(
        get_randint(),
        nochad_tree,
        bcode_meta,
        args.max_iters,
        args.num_inits,
        trans_wrap_maker,
        fit_param_list=[fit_params],
        known_params=args.known_params,
        assessor=assessor).run_worker(None)[0]

    warm_start_params = no_chad_res.get_fit_params()
    prev_branch_inners = warm_start_params["branch_len_inners"].copy()
    prev_branch_proportions = warm_start_params["branch_len_offsets_proportion"].copy()

    worker_list = []
    other_chad_parents = [
            p for p in hanging_chad.possible_parents
            if p.node_id != current_parent_id]
    num_other_parents = len(other_chad_parents)
    random_order = np.random.choice(num_other_parents, num_other_parents, replace=False)
    chosen_idxs = random_order[:args.max_chad_tune_search - 1]
    possible_chad_parents = ([
            p for p in hanging_chad.possible_parents
            if p.node_id == current_parent_id] + [other_chad_parents[idx] for idx in chosen_idxs])
    logging.info(
            "chad parent idxs considered %s (out of %d) (plus the orig one)",
            chosen_idxs,
            num_other_parents)
    for chad_par in possible_chad_parents:
        # From the no chad tree, add back the hanging chad to the designated parent
        tree_copy = nochad_tree.copy()
        for node in tree_copy.traverse():
            node.add_feature("nochad_id", node.node_id)
            # Track which nodes were originally leaves in the nochad tree
            # These nodes have nonsense values in the branch_len_inner param,
            # so we will make sure to ignore those values.
            node.add_feature("nochad_is_leaf", node.is_leaf())

            node.add_feature("nochad_resolved_multifurc", node.is_resolved_multifurcation())

        for node in tree_copy.traverse():
            if node.orig_node_id is not None and node.orig_node_id == chad_par.node_id:
                if node.is_leaf():
                    new_child = node.copy()
                    new_child.add_features(
                        orig_node_id=None,
                        node_id=None,
                        nochad_id=None)
                    node.add_child(new_child)
                new_hanging_chad = hanging_chad.node.copy()
                new_hanging_chad.add_features(
                        orig_node_id=None,
                        node_id=None,
                        nochad_id=None)
                for descendant in new_hanging_chad.get_descendants():
                    descendant.add_features(
                        orig_node_id=None,
                        node_id=None,
                        nochad_id=None)
                node.add_child(new_hanging_chad)
                break

        num_nodes = tree_copy.label_node_ids()
        assert len(tree_copy) == len(tree)

        # warm start the branch length estimates
        warm_start_params = no_chad_res.get_fit_params()
        warm_start_params["branch_len_inners"] = np.ones(num_nodes) * 1e-10
        warm_start_params["branch_len_offsets_proportion"] = np.ones(num_nodes) * 0.45 + np.random.rand(num_nodes) * 0.1
        for node in tree_copy.traverse():
            if node.nochad_id is not None:
                # Copy over existing branch length estimates -- if it matches an existing branch in the no-chad tree
                nochad_id = int(node.nochad_id)
                node_id = int(node.node_id)
                if not node.nochad_is_leaf:
                    # Only copy branch_len_inners for non-leaf nodes (from the nochad tree) because branch_len_inner values
                    # are meaningless for leaf nodes
                    warm_start_params["branch_len_inners"][node_id] = max(
                        prev_branch_inners[nochad_id] - 1e-9, 1e-10)
                if node.up is not None:
                    if not node.up.nochad_resolved_multifurc:
                        warm_start_params["branch_len_offsets_proportion"][node_id] = prev_branch_proportions[nochad_id]
                    else:
                        warm_start_params["branch_len_offsets_proportion"][node_id] = 1e-10

        trans_wrap_maker = TransitionWrapperMaker(
            tree_copy,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
        worker = LikelihoodScorer(
            get_randint(),
            tree_copy,
            bcode_meta,
            args.max_iters,
            args.num_inits,
            trans_wrap_maker,
            fit_param_list=[warm_start_params],
            known_params=args.known_params,
            assessor=assessor)
        worker_list.append(worker)

    logging.info("CHAD TUNING")
    if args.num_processes > 1 and len(worker_list) > 1:
        job_manager = SubprocessManager(
                worker_list,
                None,
                args.scratch_dir,
                args.num_processes)
        worker_results = [w[0][0] for w in job_manager.run()]
    else:
        worker_results = [w.run_worker(None)[0] for w in worker_list]

    TREE_DIST_MEASURERS = [UnrootRFDistanceMeasurer, BHVDistanceMeasurer, SPRDistanceMeasurer, InternalHeightsMeasurer]
    for tree_dist_measurer in TREE_DIST_MEASURERS:
        chad_tune_res = _create_chad_results(
            worker_results,
            possible_chad_parents,
            no_chad_res,
            hanging_chad,
            tree_dist_measurer,
            args.scratch_dir,
            args.stability_weight)

    return chad_tune_res


def _create_chad_results(
        fit_results: List[LikelihoodScorerResult],
        chad_parent_candidates: List[CellLineageTree],
        no_chad_res: LikelihoodScorerResult,
        hanging_chad: HangingChad,
        tree_dist_measurer,
        scratch_dir: str,
        weight: float = 0.5):
    """
    @return HangingChadTuneResult
    """
    assert len(fit_results) == len(chad_parent_candidates)
    assert weight >= 0 and weight <= 1
    no_chad_dist_meas = tree_dist_measurer(no_chad_res.fitted_bifurc_tree, scratch_dir)
    new_chad_results = [
        _create_chad_result(fit_res, no_chad_res, no_chad_dist_meas, hanging_chad, chad_par, weight)
        for fit_res, chad_par in zip(fit_results, chad_parent_candidates)]
    chad_tune_res = HangingChadTuneResult(no_chad_res, new_chad_results)

    for chad_res in new_chad_results:
        logging.info("Chad res: %s", str(chad_res))
        if "performance" in chad_res.fit_res.train_history[-1]:
            logging.info("  chad truth: %s", chad_res.fit_res.train_history[-1]["performance"])

    if "performance" in no_chad_res.train_history[-1]:
        assess_metric = "bhv"
        all_tree_dists = [c.fit_res.train_history[-1]["performance"][assess_metric] for c in new_chad_results]
        targ_dists = [c.fit_res.train_history[-1]["performance"]["targ"] for c in new_chad_results]
        median_tree_dist = np.median(all_tree_dists)
        tree_scores = [c.tree_score for c in new_chad_results]
        targ_scores = [c.targ_score for c in new_chad_results]
        selected_idx = np.argmax([c.score for c in new_chad_results])
        selected_chad = new_chad_results[selected_idx]
        logging.info("Best chad %s (%s)", selected_chad, no_chad_dist_meas.name)
        selected_tree_dist = all_tree_dists[selected_idx]
        logging.info(
                "Median %s: %f (%d options) (selected idx %d, better? %s, selected=%f) -- %s",
                assess_metric,
                median_tree_dist,
                len(all_tree_dists),
                selected_idx,
                median_tree_dist > selected_tree_dist,
                selected_tree_dist,
                no_chad_dist_meas.name)
        logging.info("all_%s= %s", assess_metric, all_tree_dists)
        logging.info("targ_dists= %s", targ_dists)
        logging.info("tree_%s = %s", no_chad_dist_meas.name, tree_scores)
        logging.info("targ_scores = %s", targ_scores)
        logging.info("tree_%s vs. %ss %s", no_chad_dist_meas.name, assess_metric, scipy.stats.spearmanr(tree_scores, all_tree_dists))
        logging.info("targ vs. %ss %s", assess_metric, scipy.stats.spearmanr(targ_scores, all_tree_dists))
        logging.info("tree_%s vs. targ dists %s",  no_chad_dist_meas.name, scipy.stats.spearmanr(tree_scores, targ_dists))
        logging.info("targ vs. targ dists %s", scipy.stats.spearmanr(targ_scores, targ_dists))
    return chad_tune_res


def _create_chad_result(
        new_chad_res: LikelihoodScorerResult,
        no_chad_res: LikelihoodScorerResult,
        no_chad_dist_meas: TreeDistanceMeasurer,
        hanging_chad: HangingChad,
        chad_par: CellLineageTree,
        weight: float = 0.5):
    """
    @return HangingChadResult
    """
    # Get the target lambda stability score
    new_chad_targs = new_chad_res.get_all_target_params()
    no_chad_targs = no_chad_res.get_all_target_params()
    targ_stability_score = -np.linalg.norm(new_chad_targs - no_chad_targs)

    # Get the tree stability score
    # Shrink down the tree to compare
    tree_leaf_strs = set([l.allele_events_list_str for l in no_chad_res.fitted_bifurc_tree])
    keep_leaf_ids = set()
    for leaf in new_chad_res.fitted_bifurc_tree:
        if leaf.allele_events_list_str in tree_leaf_strs:
            keep_leaf_ids.add(leaf.node_id)
    new_tree_pruned = CellLineageTree.prune_tree(new_chad_res.fitted_bifurc_tree, keep_leaf_ids)
    tree_stability_score = -no_chad_dist_meas.get_dist_multifurc_tree(
        new_tree_pruned,
        no_chad_res.orig_tree)

    return HangingChadResult(
        targ_stability_score,
        tree_stability_score,
        hanging_chad.node,
        chad_par,
        new_chad_res,
        weight)
