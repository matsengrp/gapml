import numpy as np
import scipy.stats
from typing import List, Dict, Set
import logging
import random

from allele_events import AlleleEvents
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import SubprocessManager
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from common import get_randint, get_init_target_lams
from model_assessor import ModelAssessor
import collapsed_tree
import ancestral_events_finder
from clt_likelihood_penalization import mark_target_status_to_penalize


"""
Hanging chad is our affectionate name for the inter-target cuts that have ambiguous placement
in the tree. For example, if we don't know if the 1-3 inter-target cut is a child of the root
node or a child of another allele with a 2-2 intra-target cut.
"""


class HangingChadResult:
    def __init__(
            self,
            score: float,
            chad_node_id: int,
            full_chad_tree: CellLineageTree,
            fit_res: LikelihoodScorerResult):
        """
        @param score: higher score means "better" place for hanging chad
        @param chad_node: the hanging chad we need to find a parent for
        @param parent_node: the candidate parent node for our hanging chad
        @param full_chad_tree: the entire tree with the hanging chad placed under that parent node
        @param fit_res: a list of fitting results when we placed the hanging chad under that candidate parent
        """
        self.score = score
        self.full_chad_tree = full_chad_tree
        self.chad_node = full_chad_tree.search_nodes(node_id=chad_node_id)[0]
        self.parent_node = self.chad_node.up

        self.fit_res = fit_res
        train_hist_last = fit_res.train_history[-1]
        self.true_performance = train_hist_last['performance'] if 'performance' in train_hist_last else None

    def __str__(self):
        return "%s=>%s (score=%f)" % (
                self.parent_node.anc_state_list_str,
                self.chad_node.allele_events_list_str,
                self.score)


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
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        if self.no_chad_res is not None:
            no_chad_fit_params = self.no_chad_res.get_fit_params()
            fit_params['dist_to_half_pen_param'] = no_chad_fit_params['dist_to_half_pen_param']
            fit_params['log_barr_pen_param'] = no_chad_fit_params['log_barr_pen_param']

        orig_tree = best_chad.full_chad_tree.copy()
        collapsed_tree._remove_single_child_unobs_nodes(orig_tree)

        return orig_tree, fit_params, best_fit_res


class HangingChad:
    def __init__(
            self,
            node: CellLineageTree,
            nochad_tree: CellLineageTree,
            possible_full_trees: List[CellLineageTree]):
        """
        @param node: the hanging chad
        @param possible_full_trees: first tree is the original tree
        """
        self.node = node
        self.nochad_tree = nochad_tree
        self.possible_full_trees = possible_full_trees
        self.num_possible_trees = len(possible_full_trees)
        # Kind of an id for this hanging chad
        self.psuedo_id = str([str(a) for a in node.anc_state_list])

        self.chad_ids = set([node.node_id for node in self.node.traverse()])

        self.nochad_leaf = dict()
        self.nochad_unresolved_multifurc = dict()
        for node in nochad_tree.traverse():
            self.nochad_leaf[node.node_id] = node.is_leaf()
            self.nochad_unresolved_multifurc[node.node_id] = not node.is_resolved_multifurcation()

    def make_single_leaf_rand_trees(self):
        """
        @return List[CellLineageTree] -- takes a random leaf of the hanging chad and only keeps that
                    in each of the possible trees, drops all other leaves of the hanging chad
        """
        chad_leaf_ids = [leaf.node_id for leaf in self.node]
        random_leaf_id = random.choice(chad_leaf_ids)
        num_chad_leaves = len(chad_leaf_ids)
        single_leaf_rand_trees = []
        for idx, full_tree in enumerate(self.possible_full_trees):
            single_leaf_tree = full_tree.copy()
            chad_in_tree = single_leaf_tree.search_nodes(node_id=self.node.node_id)[0]
            implicit_nodes = single_leaf_tree.search_nodes(node_id=None)

            for node in single_leaf_tree.traverse('postorder'):
                node.add_feature(
                    'nochad_id',
                    node.node_id if node.node_id not in self.chad_ids else None)

            if num_chad_leaves > 1:
                # Prune the tree
                old_parent = chad_in_tree.up
                rand_leaf_in_tree = single_leaf_tree.search_nodes(node_id=random_leaf_id)[0]
                chad_in_tree.detach()
                old_parent.add_child(rand_leaf_in_tree)
                rand_leaf_in_tree.add_feature("ignore_penalty", True)
            else:
                chad_in_tree.add_feature("ignore_penalty", True)

            single_leaf_tree.label_node_ids()
            assert sum([leaf.nochad_id is None for leaf in single_leaf_tree]) == 1

            if len(implicit_nodes) >= 1:
                assert len(implicit_nodes) == 1
                implicit_node = implicit_nodes[0]
                assert implicit_node.nochad_id is None
                assert len(implicit_node.get_children()) == 2
                for child in implicit_node.get_children():
                    if child.nochad_id is not None:
                        child.add_feature('spine_children', [])
                        implicit_node.add_features(
                            implicit_child=child,
                            spine_children=[implicit_node.node_id, child.node_id])
                        break
                assert implicit_node.implicit_child is not None

            single_leaf_rand_trees.append({
                "full": full_tree,
                "single_leaf": single_leaf_tree})
        return single_leaf_rand_trees

    def __str__(self):
        chad_parent_strs = []
        for full_tree in self.possible_full_trees:
            chad_in_tree = full_tree.search_nodes(node_id=self.node.node_id)[0]
            chad_parent_strs.append("%d,%s" % (
                chad_in_tree.up.node_id if chad_in_tree.up.is_root() else chad_in_tree.up.up.node_id,
                chad_in_tree.up.anc_state_list_str))
        return "%s: %d leaves, %d possibilities: %s" % (
            self.node.anc_state_list_str,
            len(self.node),
            self.num_possible_trees,
            chad_parent_strs)


def _get_chad_possibilities(
        chad_id: int,
        tree: CellLineageTree,
        parsimony_score: int,
        bcode_meta: BarcodeMetadata,
        max_possible_trees: int = None,
        branch_len_attaches: bool = True):
    """
    @return List[CellLineageTree] that are equally parsimonious trees after putting chad on various
            branches and nodes
    """
    chad = tree.search_nodes(node_id=chad_id)[0]
    assert not hasattr(chad, "nochad_id")

    # Create no chad tree -- by detaching chad
    chad_orig_parent = chad.up
    chad.detach()
    chad_orig_parent_unifurc = len(chad_orig_parent.get_children()) == 1
    num_nochad_nodes = tree.label_node_ids()
    for idx, node in enumerate(chad.traverse()):
        node.node_id = num_nochad_nodes + idx
    num_labelled_nodes = num_nochad_nodes + idx + 1
    chad_copy = chad.copy()
    nochad_tree = tree.copy()

    # First create original tree
    chad_orig_parent.add_child(chad)
    orig_tree = tree.copy()

    # sanity check that original tree is equally parsimonious
    ancestral_events_finder.annotate_ancestral_states(orig_tree, bcode_meta)
    new_pars_score = ancestral_events_finder.get_parsimony_score(orig_tree)
    if new_pars_score < parsimony_score:
        logging.info("orig has prob %d< %d", new_pars_score, parsimony_score)
        logging.info(tree.get_ascii(attributes=['anc_state_list_str']))
        logging.info(tree.get_ascii(attributes=['dist']))
    assert new_pars_score == parsimony_score

    # Now find all the possible trees we can make by hanging the chad elsewhere
    possible_trees = []
    # Only consider nodes that are not descendnats of the chad
    all_nodes = [node for node in tree.traverse() if node.node_id < num_nochad_nodes]
    # First consider adding chad to existing nodes
    for node in all_nodes:
        if max_possible_trees is not None and len(possible_trees) >= max_possible_trees - 1:
            break
        if node.is_leaf() or node.node_id == chad_orig_parent.node_id:
            continue

        chad.detach()
        node.add_child(chad)

        ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
        new_pars_score = ancestral_events_finder.get_parsimony_score(tree)
        if new_pars_score < parsimony_score:
            logging.info("We beat MIX (attach to multifurc): %d< %d", new_pars_score, parsimony_score)
            logging.info(tree.get_ascii(attributes=['anc_state_list_str']))
            logging.info(tree.get_ascii(attributes=['dist']))
        #assert new_pars_score >= parsimony_score
        can_collapse_tree = any([other_node.dist == 0 for other_node in tree.get_descendants() if not other_node.is_leaf()])
        if new_pars_score == parsimony_score and not can_collapse_tree:
            tree_copy = tree.copy()
            possible_trees.append(tree_copy)

    if branch_len_attaches:
        logging.info('find branch length attachements')
        # Consider adding chad to the middle of the existing edges
        for node in all_nodes:
            if max_possible_trees is not None and len(possible_trees) >= max_possible_trees - 1:
                break
            if node.is_root() or (chad_orig_parent_unifurc and node.node_id == chad_orig_parent.node_id):
                continue

            chad.detach()

            # Consider adding chad to an edge, so parent -- new_node -- node, chad
            new_node = CellLineageTree(
                    allele_events_list=[
                        AlleleEvents([], allele_evts.num_targets)
                        for allele_evts in node.allele_events_list])
            # new node will have the biggest node id
            new_node.add_features(node_id=None)
            node.up.add_child(new_node)
            node.detach()
            new_node.add_child(node)
            new_node.add_child(chad)

            ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
            new_pars_score = ancestral_events_finder.get_parsimony_score(tree)
            if new_pars_score < parsimony_score:
                logging.info("we beat MIX (attach in middle): %d< %d", new_pars_score, parsimony_score)
                logging.info(tree.get_ascii(attributes=['anc_state_list_str']))
                logging.info(tree.get_ascii(attributes=['dist']))

            #assert new_pars_score >= parsimony_score

            # Then this can be further collapsed and we might as well add the node to
            # a multif/bifurcation. Should have been accomplished previously.
            can_collapse_tree = any([other_node.dist == 0 for other_node in tree.get_descendants() if not other_node.is_leaf()])
            if not can_collapse_tree and new_pars_score == parsimony_score:
                all_node_ids = set([node.node_id for node in tree.traverse() if node.node_id is not None])
                assert all_node_ids == set(list(range(num_labelled_nodes)))
                num_new_nodes = len([node for node in tree.traverse() if node.node_id is None])
                assert num_new_nodes == 1

                possible_trees.append(tree.copy())

            # Now undo all the things we just did to the tree
            new_node.delete()

    random.shuffle(possible_trees)
    return HangingChad(
        chad_copy,
        nochad_tree,
        [orig_tree] + possible_trees)

def _preprocess_tree_for_chad_finding(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
    parsimony_score = ancestral_events_finder.get_parsimony_score(tree)
    tree = collapsed_tree.collapse_zero_lens(tree)
    can_collapse_tree = any([other_node.dist == 0 for other_node in tree.get_descendants() if not other_node.is_leaf()])
    logging.info(tree.get_ascii(attributes=['anc_state_list_str']))
    logging.info(tree.get_ascii(attributes=['dist']))
    assert not can_collapse_tree

    return tree, parsimony_score


def get_random_chad(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        exclude_chads: Set = set(),
        exclude_chad_func = None,
        branch_len_attaches: bool = True):
    """
    @return a randomly chosen HangingChad, also a set of the recently seen chads (indexed by psuedo id)
    """
    tree, parsimony_score = _preprocess_tree_for_chad_finding(tree, bcode_meta)

    descendants = tree.get_descendants()
    random.shuffle(descendants)
    for node in descendants:
        if exclude_chad_func is not None and exclude_chad_func(node) in exclude_chads:
            continue

        has_masking_cuts = any([sg.min_target + 1 < sg.max_target for anc_state in node.anc_state_list for sg in anc_state.get_singletons()])
        if not has_masking_cuts:
            continue

        tree_copy = tree.copy()
        hanging_chad = _get_chad_possibilities(
                node.node_id,
                tree_copy,
                parsimony_score,
                bcode_meta,
                branch_len_attaches=branch_len_attaches)
        if hanging_chad.num_possible_trees > 1:
            if exclude_chad_func is not None:
                exclude_chads.add(exclude_chad_func(hanging_chad.node))
            logging.info("ran %s", str(hanging_chad))

            possible_trees = hanging_chad.possible_full_trees
            chad = hanging_chad.node
            logging.info("num possible pars trees %d", len(possible_trees) + 1)
            logging.info("chad %d %s", chad.node_id, chad.allele_events_list_str)
            for t_idx, p_tree in enumerate(possible_trees):
                logging.info("chad %d %s", chad.node_id, chad.allele_events_list_str)
                logging.info(p_tree.get_ascii(attributes=['anc_state_list_str']))
                logging.info(p_tree.get_ascii(attributes=['dist']))

            return hanging_chad, exclude_chads

    if len(exclude_chads):
        # We still haven't found our chad friend apparently...
        logging.info("need to start over chad set")
        # Did we exclude all of them?
        for node in descendants:
            if node.anc_state_list_str not in exclude_chads:
                continue

            has_masking_cuts = any([sg.min_target + 1 < sg.max_target for anc_state in node.anc_state_list for sg in anc_state.get_singletons()])
            if not has_masking_cuts:
                continue

            tree_copy = tree.copy()
            hanging_chad = _get_chad_possibilities(
                    node.node_id,
                    tree_copy,
                    parsimony_score,
                    bcode_meta,
                    branch_len_attaches=branch_len_attaches)
            if hanging_chad.num_possible_trees > 1:
                if exclude_chad_func is not None:
                    exclude_chads = set([exclude_chad_func(hanging_chad.node)])
                return hanging_chad, exclude_chads
    return None, None


def get_all_chads(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        max_possible_trees: int = None):
    """
    @param max_possible_trees: stop once you find at least `max_possible_trees` in a chad
    @return List[HangingChad] all hanging chads in the tree
    """
    tree, parsimony_score = _preprocess_tree_for_chad_finding(tree, bcode_meta)

    hanging_chads = []
    for node in tree.get_descendants():
        has_masking_cuts = any([sg.min_target + 1 < sg.max_target for anc_state in node.anc_state_list for sg in anc_state.get_singletons()])
        if not has_masking_cuts:
            continue

        tree_copy = tree.copy()
        hanging_chad = _get_chad_possibilities(
                node.node_id,
                tree_copy,
                parsimony_score,
                bcode_meta,
                max_possible_trees=max_possible_trees)
        if hanging_chad.num_possible_trees > 1:
            hanging_chads.append(hanging_chad)
    return hanging_chads


def _fit_nochad_result(
        hanging_chad: HangingChad,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor = None):
    """
    @param hanging_chad: the hanging chad to remove from the tree
    @param tree: the original tree

    @return LikelihoodScorerResult, the node_id of the current parent node of the hanging chad
    """
    nochad_tree = hanging_chad.nochad_tree
    logging.info("no chad tree leaves %d", len(nochad_tree))

    if not args.known_params.target_lams:
        fit_params['target_lams'] = get_init_target_lams(fit_params['target_lams'].size)
    fit_params.pop('branch_len_inners', None)
    fit_params.pop('branch_len_offsets_proportion', None)
    fit_params['conv_thres'] = 5 * 1e-4

    # Now fit the tree without the hanging chad
    trans_wrap_maker = TransitionWrapperMaker(
        nochad_tree,
        bcode_meta,
        args.max_extra_steps,
        args.max_sum_states)
    hanging_chad.pen_anc_state = dict()
    mark_target_status_to_penalize(nochad_tree)
    for node in nochad_tree.get_descendants():
        hanging_chad.pen_anc_state[node.node_id] = node.pen_targ_stat

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
    return no_chad_res


def _create_warm_start_fit_params(
        hanging_chad: HangingChad,
        nochad_res: LikelihoodScorerResult,
        new_chad_tree: CellLineageTree):
    """
    Assemble the `fit_param` and `KnownModelParam` for fitting the tree with the hanging chad
    This does a warm start -- it copies over all model parameters and branch lengths
    @return Dict, KnownModelParam
    """
    num_nodes = new_chad_tree.get_num_nodes()

    fit_params = nochad_res.get_fit_params()

    prev_branch_inners = fit_params["branch_len_inners"].copy()
    prev_branch_proportions = fit_params["branch_len_offsets_proportion"].copy()

    fit_params['conv_thres'] = 5 * 1e-7
    fit_params["branch_len_inners"] = np.ones(num_nodes)
    fit_params["branch_len_offsets_proportion"] = np.ones(num_nodes) * 0.45 + np.random.rand(num_nodes) * 0.1
    for node in new_chad_tree.traverse():
        if node.nochad_id is None:
            # This is completely new node. No copying to do
            continue

        if not hanging_chad.nochad_unresolved_multifurc[node.nochad_id]:
            # If this wasnt a multifurc and has suddenly become one,
            # we will just force it to be a true resolved multifurcation.
            node.resolved_multifurcation = True

        # Make sure no nodes in the tree used to be a leaf and are no longer a leaf
        assert hanging_chad.nochad_leaf[node.nochad_id] == node.is_leaf()
        fit_params["branch_len_inners"][node.node_id] = prev_branch_inners[node.nochad_id]
        fit_params["branch_len_offsets_proportion"][node.node_id] = prev_branch_proportions[node.nochad_id]

    return fit_params


def tune(
        hanging_chad: HangingChad,
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor = None,
        print_assess_metric: str = "bhv"):
    """
    Tune the given hanging chad
    @return HangingChadTuneResult
    """
    assert hanging_chad.num_possible_trees > 1
    no_chad_res = _fit_nochad_result(
        hanging_chad,
        bcode_meta,
        args,
        fit_params,
        assessor=assessor)

    worker_list = []
    # Pick a random leaf from the hanging chad -- do not use the entire hanging chad
    # This is because the entire hanging chad might have multiple leaves and their
    # branch length assignment is ambigious.
    new_chad_tree_dicts = hanging_chad.make_single_leaf_rand_trees()[:args.max_chad_tune_search]
    for parent_idx, new_chad_tree_dict in enumerate(new_chad_tree_dicts):
        new_chad_tree = new_chad_tree_dict["single_leaf"]

        warm_start_fit_params = _create_warm_start_fit_params(
            hanging_chad,
            no_chad_res,
            new_chad_tree)

        # Mark the target statuses to penalize -- these are adopted from the nochad tree
        for node in new_chad_tree.get_descendants():
            # Recall that we do not penalize the hanging chad
            if node.nochad_id is not None:
                node.add_feature('pen_targ_stat',
                    hanging_chad.pen_anc_state[node.nochad_id])
                if node.up.nochad_id is None:
                    # If the parent node is the implicit node,
                    # we should have the implicit node get penalized.
                    # The child of the implicit node will not be.
                    node.up.add_feature('pen_targ_stat',
                        hanging_chad.pen_anc_state[node.nochad_id])

        trans_wrap_maker = TransitionWrapperMaker(
            new_chad_tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
        worker = LikelihoodScorer(
            get_randint(),
            new_chad_tree,
            bcode_meta,
            args.max_iters,
            args.num_inits,
            trans_wrap_maker,
            fit_param_list=[warm_start_fit_params],
            known_params=args.known_params,
            assessor=assessor,
            name="chad-tuning%d" % parent_idx)
        worker_list.append(worker)

    # Actually fit the results
    logging.info("CHAD TUNING")
    job_manager = SubprocessManager(
            worker_list,
            None,
            args.scratch_dir,
            args.num_processes)
    worker_results = [w[0][0] for w in job_manager.run()]

    # Aggregate the results
    chad_tune_res = _create_chad_results(
        worker_results,
        new_chad_tree_dicts,
        no_chad_res,
        hanging_chad,
        args.scratch_dir)

    # Do some logging on the results
    for chad_res in chad_tune_res.new_chad_results:
        logging.info("Chad res: %s", str(chad_res))
        if assessor is not None:
            logging.info("  chad truth: %s", chad_res.true_performance)

    if assessor is not None:
        new_chad_results = chad_tune_res.new_chad_results
        all_tree_dists = [c.true_performance[print_assess_metric] for c in new_chad_results]
        median_tree_dist = np.median(all_tree_dists)
        scores = [c.score for c in new_chad_results]
        selected_idx = np.argmax([c.score for c in new_chad_results])
        selected_chad = new_chad_results[selected_idx]
        logging.info("Best chad %s", selected_chad)
        selected_tree_dist = all_tree_dists[selected_idx]
        logging.info(
                "Median %s: %f (%d options) (selected idx %d, better? %s, selected=%f)",
                print_assess_metric,
                median_tree_dist,
                len(all_tree_dists),
                selected_idx,
                median_tree_dist > selected_tree_dist,
                selected_tree_dist)
        logging.info("bhv range = %f to %f", np.min(all_tree_dists), np.max(all_tree_dists))
        logging.info("all_%s= %s", print_assess_metric, all_tree_dists)
        logging.info("scores = %s", scores)
        logging.info("scores vs. %ss %s", print_assess_metric, scipy.stats.spearmanr(scores, all_tree_dists))

    return chad_tune_res


def _create_chad_results(
        fit_results: List[LikelihoodScorerResult],
        new_chad_tree_dicts: List[Dict],
        no_chad_res: LikelihoodScorerResult,
        hanging_chad: HangingChad,
        scratch_dir: str):
    """
    Process the results and aggregate them in a HangingChadTuneResult

    @return HangingChadTuneResult
    """
    assert len(fit_results) == len(new_chad_tree_dicts)

    new_chad_results = [
        HangingChadResult(
            fit_res.pen_log_lik[0],
            hanging_chad.node.node_id,
            chad_tree_dict["full"],
            fit_res)
        for fit_res, chad_tree_dict in zip(fit_results, new_chad_tree_dicts)]
    return HangingChadTuneResult(no_chad_res, new_chad_results)
