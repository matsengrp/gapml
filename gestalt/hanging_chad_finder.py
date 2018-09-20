import numpy as np
import scipy.stats
from typing import List, Dict
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
from optim_settings import KnownModelParams
import collapsed_tree
import ancestral_events_finder
from constants import PERTURB_ZERO


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
        no_chad_fit_params = self.no_chad_res.get_fit_params()
        # Popping branch length estimates right now because i dont know
        # how to warm start using these estimates...?
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
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
        self.psuedo_id = node.anc_state_list_str

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
        random_leaf_id = random.choice([leaf.node_id for leaf in self.node])
        single_leaf_rand_trees = []
        for full_tree in self.possible_full_trees:
            single_leaf_tree = full_tree.copy()
            chad_in_tree = single_leaf_tree.search_nodes(node_id=self.node.node_id)[0]
            num_chad_leaves = len(chad_in_tree)

            # Prune the tree
            for child in chad_in_tree.get_children():
                if sum([leaf.node_id == random_leaf_id for leaf in child]) == 0:
                    child.detach()

            # Handle the case where the chad was added to an edge
            if chad_in_tree.up.node_id is None:
                grandpa = chad_in_tree.up.up
                implicit_node = CellLineageTree(
                        allele_events_list=[
                            AlleleEvents([], allele_evts.num_targets)
                            for allele_evts in self.node.allele_events_list])
                implicit_node.add_features(node_id=None)
                chad_in_tree.up.detach()
                grandpa.add_child(implicit_node)
                implicit_node.add_child(chad_in_tree.up)

            for node in single_leaf_tree.traverse():
                if node.node_id not in self.chad_ids:
                    node.add_feature("nochad_id", node.node_id)
                else:
                    node.add_feature("nochad_id", None)

            if num_chad_leaves > 1:
                # There is a unifurcation since we pruned away the other leaves
                # delete the intervening unifurcation
                chad_in_tree.delete()
            single_leaf_tree.label_node_ids()

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
        scratch_dir):
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
    assert new_pars_score == parsimony_score

    # Now find all the possible trees we can make by hanging the chad elsewhere
    possible_trees = []
    # Only consider nodes that are not descendnats of the chad
    all_nodes = [node for node in tree.traverse() if node.node_id < num_nochad_nodes]
    # First consider adding chad to existing nodes
    for node in all_nodes:
        if node.is_leaf() or node.node_id == chad_orig_parent.node_id:
            continue

        chad.detach()
        node.add_child(chad)

        ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
        new_pars_score = ancestral_events_finder.get_parsimony_score(tree)
        assert new_pars_score >= parsimony_score
        can_collapse_tree = any([other_node.dist == 0 for other_node in tree.get_descendants() if not other_node.is_leaf()])
        if new_pars_score == parsimony_score and not can_collapse_tree:
            tree_copy = tree.copy()
            possible_trees.append(tree_copy)

    # Consider adding chad to the middle of the existing edges
    for node in all_nodes:
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
        assert new_pars_score >= parsimony_score

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

    #logging.info("num possible pars trees %d", len(possible_trees) + 1)
    #logging.info("chad %d %s", chad.node_id, chad.allele_events_list_str)
    #if len(possible_trees) + 1 > 1:
    #    logging.info(orig_tree.get_ascii(attributes=['allele_events_list_str']))
    #    logging.info(orig_tree.get_ascii(attributes=['dist']))
    #    for t_idx, p_tree in enumerate(possible_trees):
    #        logging.info("chad %d %s", chad.node_id, chad.allele_events_list_str)
    #        logging.info(p_tree.get_ascii(attributes=['anc_state_list_str']))
    #        logging.info(p_tree.get_ascii(attributes=['dist']))

    random.shuffle(possible_trees)
    return HangingChad(
        chad_copy,
        nochad_tree,
        [orig_tree] + possible_trees)


def get_all_chads(tree: CellLineageTree, bcode_meta: BarcodeMetadata, scratch_dir: str):
    hanging_chads = []
    collapsed_tree._remove_single_child_unobs_nodes(tree)
    ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
    parsimony_score = ancestral_events_finder.get_parsimony_score(tree)
    can_collapse_tree = any([other_node.dist == 0 for other_node in tree.get_descendants() if not other_node.is_leaf()])
    logging.info(tree.get_ascii(attributes=['anc_state_list_str']))
    logging.info(tree.get_ascii(attributes=['dist']))
    assert not can_collapse_tree

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
                scratch_dir)
        if hanging_chad.num_possible_trees > 1:
            hanging_chads.append(hanging_chad)
    return hanging_chads


def _fit_nochad_result(
        nochad_tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        assessor: ModelAssessor = None):
    """
    @param hanging_chad: the hanging chad to remove from the tree
    @param tree: the original tree

    @return LikelihoodScorerResult, the node_id of the current parent node of the hanging chad
    """
    logging.info("no chad tree leaves %d", len(nochad_tree))

    fit_params['target_lams'] = get_init_target_lams(fit_params['target_lams'].size)
    fit_params.pop('branch_len_inners', None)
    fit_params.pop('branch_len_offsets_proportion', None)

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
    return no_chad_res


def _create_warm_start_fit_params(
        hanging_chad: HangingChad,
        no_chad_res: LikelihoodScorerResult,
        new_chad_tree: CellLineageTree):
    """
    Assemble the `fit_param` and `KnownModelParam` for fitting the tree with the hanging chad
    This does a warm start -- it copies over all model parameters and branch lengths
    @return Dict, KnownModelParam
    """
    num_nodes = new_chad_tree.get_num_nodes()

    fit_params = no_chad_res.get_fit_params()
    prev_branch_inners = fit_params["branch_len_inners"].copy()
    prev_branch_proportions = fit_params["branch_len_offsets_proportion"].copy()

    fit_params['dist_to_half_pen_param'] = 0
    branch_len_inners_mask = np.zeros(num_nodes, dtype=bool)
    branch_len_offsets_proportion_mask = np.zeros(num_nodes, dtype=bool)
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

        if node.is_root() or node.up.nochad_id is not None:
            # Parent node and this node are both in the no chad tre
            # therefore the branch length is completely specified
            # Copy over existing branch length estimates -- it matches an existing branch in the no-chad tree
            branch_len_inners_mask[node.node_id] = True
            branch_len_offsets_proportion_mask[node.node_id] = True

            assert hanging_chad.nochad_leaf[node.nochad_id] == node.is_leaf()
            fit_params["branch_len_inners"][node.node_id] = prev_branch_inners[node.nochad_id]

            fit_params["branch_len_offsets_proportion"][node.node_id] = prev_branch_proportions[node.nochad_id]
        elif node.up.nochad_id is None:
            # This is an implicit node -- we know how much the distance is between grandparent
            # and current node, but don't know where to place the parent node. This happens when we insert the hanging
            # chad along a branch instead of at an existing node
            assert node.up.up.nochad_id is None
            # Only mark the node's inner var as known. We don't know it's proportion. The proportin will specify the location
            # of the implicit parent node.
            branch_len_inners_mask[node.node_id] = True
            # mark the implicit nodes having known variables since they are defined via other params.
            # these parameters are completely ignored
            branch_len_inners_mask[node.up.node_id] = True
            branch_len_offsets_proportion_mask[node.up.node_id] = True
            # also mark the grandparent having known branch length params (grandparent nodes are newly introduced
            # to deal with fixing the offset along the spine) -- we need to calculate the offset values for the
            # grandparent
            branch_len_inners_mask[node.up.up.node_id] = True
            branch_len_offsets_proportion_mask[node.up.up.node_id] = True

            # Get the branch length inner of the node -- we do this convoluted thing in case the node in
            # question is actually a leaf. (then its branch_len_inner value is bogus)
            par_in_nochad_tree = node.up.up.up.nochad_id
            node_up_dist_to_root = no_chad_res.train_history[-1]['dist_to_roots'][par_in_nochad_tree]
            node_dist_to_root = no_chad_res.train_history[-1]['dist_to_roots'][node.nochad_id]
            node_branch_inner = node_dist_to_root - node_up_dist_to_root

            if hanging_chad.nochad_unresolved_multifurc[par_in_nochad_tree]:
                # If our node is a child of a multifurcation in the nochad tree, then we need to fix the node's offset
                # from the multifurcation and fix up the inner var value for this node.
                proportion = prev_branch_proportions[node.nochad_id]
                # Fixing the offset from the grandparent located on the spine
                fit_params["branch_len_inners"][node.node_id] = node_branch_inner * (1 - proportion)
                # Fixing the grandparent's offset from the multifurc
                fit_params["branch_len_inners"][node.up.up.node_id] = node_branch_inner * proportion
                fit_params["branch_len_offsets_proportion"][node.up.up.node_id] = 1 - PERTURB_ZERO
            else:
                # The curr node is a child of a bifurcation. The grandparent node is placed zero dist from the parent of this node
                # from the original nochad tree.
                fit_params["branch_len_inners"][node.node_id] = node_branch_inner * (1 - PERTURB_ZERO)
                fit_params["branch_len_inners"][node.up.up.node_id] = PERTURB_ZERO

    assert np.sum(np.logical_not(branch_len_inners_mask)) > 0 and np.sum(np.logical_not(branch_len_offsets_proportion_mask)) > 0
    known_params = KnownModelParams(
        target_lams=True,
        branch_lens=True,
        branch_len_inners=branch_len_inners_mask,
        branch_len_offsets_proportion=branch_len_offsets_proportion_mask,
        tot_time=True,
        indel_params=True)

    return fit_params, known_params


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

    # Fit the nochad tree
    no_chad_res = _fit_nochad_result(
        hanging_chad.nochad_tree,
        bcode_meta,
        args,
        fit_params,
        assessor=assessor)

    worker_list = []
    new_chad_tree_dicts = hanging_chad.make_single_leaf_rand_trees()[:args.max_chad_tune_search]
    for parent_idx, new_chad_tree_dict in enumerate(new_chad_tree_dicts):
        new_chad_tree = new_chad_tree_dict["single_leaf"]

        # Pick a random leaf from the hanging chad -- do not use the entire hanging chad
        # This is because the entire hanging chad might have multiple leaves and their
        # branch length assignment is ambigious.
        warm_start_fit_params, warm_start_known_params = _create_warm_start_fit_params(
            hanging_chad,
            no_chad_res,
            new_chad_tree)

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
            known_params=warm_start_known_params,
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
            fit_res.log_lik[0],
            hanging_chad.node.node_id,
            chad_tree_dict["full"],
            fit_res)
        for fit_res, chad_tree_dict in zip(fit_results, new_chad_tree_dicts)]
    return HangingChadTuneResult(no_chad_res, new_chad_results)
