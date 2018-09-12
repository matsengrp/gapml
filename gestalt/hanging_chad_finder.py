import numpy as np
import scipy.stats
import itertools
from typing import List, Dict
import logging
import random

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import SubprocessManager
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from common import get_randint, get_init_target_lams
from model_assessor import ModelAssessor
from optim_settings import KnownModelParams
import collapsed_tree


"""
Hanging chad is our affectionate name for the inter-target cuts that have ambiguous placement
in the tree. For example, if we don't know if the 1-3 inter-target cut is a child of the root
node or a child of another allele with a 2-2 intra-target cut.
"""


class HangingChadResult:
    def __init__(
            self,
            score: float,
            chad_node: CellLineageTree,
            parent_node: CellLineageTree,
            full_chad_tree: CellLineageTree,
            fit_res: List[LikelihoodScorerResult]):
        """
        @param score: higher score means "better" place for hanging chad
        @param chad_node: the hanging chad we need to find a parent for
        @param parent_node: the candidate parent node for our hanging chad
        @param full_chad_tree: the entire tree with the hanging chad placed under that parent node
        @param fit_res: a list of fitting results when we placed the hanging chad under that candidate parent
        """
        assert len(fit_res) == 1
        self.score = score
        self.chad_node = chad_node
        self.parent_node = parent_node
        self.full_chad_tree = full_chad_tree
        self.fit_res = fit_res
        train_hist = fit_res[0].train_history[-1]
        self.true_performance = train_hist['performance'] if 'performance' in train_hist else None

    def __str__(self):
        return "%s=>%s (score=%f)" % (
                self.parent_node.allele_events_list_str,
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
        best_fit_res = best_chad.fit_res[-1]

        # TODO: how do we warm start using previous branch length estimates?
        fit_params = best_fit_res.get_fit_params()
        # Popping branch length estimates right now because i dont know
        # how to warm start using these estimates...?
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)

        orig_tree = best_chad.full_chad_tree.copy()
        collapsed_tree._remove_single_child_unobs_nodes(orig_tree)

        return orig_tree, fit_params, best_fit_res


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

    def get_rand_leaf_node(self):
        if len(self.node) == 1:
            return self.node
        else:
            return random.choice([leaf for leaf in self.node])

    def __str__(self):
        return "%s<=%s (%d leaves)" % (
            self.node.allele_events_list_str,
            [p.allele_events_list_str for p in self.possible_parents],
            len(self.node))

def _get_parsimony_score_list(node_evts, node_up_evts):
    """
    Gets the parsimony score on a single potential branch
    @param node: a potential child node
    @param up_node: a potential parent node
    @return parsimony score, returns None if this branch is impossible
    """
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


def _get_parsimony_score(node: CellLineageTree, up_node: CellLineageTree):
    """
    Gets the parsimony score on a single potential branch
    @param node: a potential child node
    @param up_node: a potential parent node
    @return parsimony score, returns None if this branch is impossible
    """
    node_evts = [set(allele_evts.events) for allele_evts in node.allele_events_list]
    node_up_evts = [set(allele_evts.events) for allele_evts in up_node.allele_events_list]
    return _get_parsimony_score_list(node_evts, node_up_evts)

def get_all_chads(tree: CellLineageTree):
    """
    Find the hanging chads in the tree without increasing the parsimony score
    this finds them ALL -- but introduces ghost nodes... need to rewrite probably

    @param tree: the tree that we are supposed to find hanging chads in
    @return List[HangingChad]
    """
    old_len = len(tree)
    for node in tree.get_descendants("preorder"):
        if node.node_id is not None:
            pars_score = _get_parsimony_score(node, node.up)
            if pars_score > 1:
                unifurc_parent = CellLineageTree(
                        node.allele_list,
                        node.allele_events_list)
                unifurc_parent.allele_events_list_str = None
                unifurc_parent.add_feature("node_id", None)
                node.up.add_child(unifurc_parent)
                node.detach()
                unifurc_parent.add_child(node)
    print(len(tree), old_len)
    assert len(tree) == old_len

    hanging_chads = []
    for node in tree.traverse("preorder"):
        if node.is_root() or node.node_id is None:
            continue

        # Get node's parsimony score
        parsimony_score = _get_parsimony_score(node, node.up)
        assert parsimony_score is not None

        # Determine if there is another place in the tree where we can preserve
        # parsimony score
        # it is possible that there is a node omitted in the parsimony tree that
        # this node can be a child of
        # Note: we are using a dictionary and preorder traversal so we don't consider
        # hanging the chad off of two nodes with the same allele_events_list_str.
        # In particular, we want to hang off of the earliest occurance of the allele_events_list_str
        # since that is the multifurc location. The other node with the same allele_events_list_str
        # is going to also dangle off the same multifurc so it is pointless to consider
        # hanging the chad off of this other node. Instead we handle that case by the continuous
        # topology tuner
        # TODO: spaggheti code
        possible_parents = {}
        for potential_par_node in tree.traverse("preorder"):
            if potential_par_node.node_id is None:
                child_node = potential_par_node.get_children()[0]
                if child_node.allele_events_list_str in possible_parents:
                    continue

                potential_score = _get_parsimony_score(node, child_node)
                if potential_score == 0 or parsimony_score == 0:
                    continue

                if potential_score is not None and potential_score == parsimony_score:
                    possible_parents[child_node.allele_events_list_str] = potential_par_node
                    # Found a parent. dont need unifurc parent
                    continue

                num_subevents = [range(len(evts.events)) for evts in potential_par_node.allele_events_list]
                num_all_events = sum([len(evts.events) for evts in potential_par_node.allele_events_list])
                for subevent_tup in itertools.product(*num_subevents):
                    event_combos = []
                    for event_count, all_evts in zip(subevent_tup, potential_par_node.allele_events_list):
                        event_combos.append([
                            list(s) for s in itertools.combinations(all_evts.events, event_count)])

                    possible_unifurc_states = itertools.product(*event_combos)
                    for unifurc_state in possible_unifurc_states:
                        unifurc_state = [set(s) for s in unifurc_state]
                        tot_unifurc_events = sum([
                            len(events) for events in unifurc_state])
                        if tot_unifurc_events == num_all_events:
                            continue
                        node_evts = [set(allele_evts.events) for allele_evts in node.allele_events_list]
                        grand_evts = [set(allele_evts.events) for allele_evts in potential_par_node.up.allele_events_list]
                        pars_state_grand = _get_parsimony_score_list(
                                unifurc_state,
                                grand_evts)
                        if pars_state_grand is None or pars_state_grand == 0:
                            continue
                        pars_state = _get_parsimony_score_list(node_evts, unifurc_state)
                        if pars_state is None:
                            continue
                        if pars_state == parsimony_score:
                            unifurc_key = str([str(s) for s in unifurc_state])
                            possible_parents[unifurc_key] = potential_par_node
                            print("found one....", parsimony_score, potential_score, child_node.allele_events_list_str)
                            print("found one....", parsimony_score, unifurc_state, node.allele_events_list_str)
            else:
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

    logging.info("total number of hanging chads %d", len(hanging_chads))
    for c in hanging_chads:
        logging.info("Chad %s", c)
    return hanging_chads


def get_chads(tree: CellLineageTree):
    """
    Find the hanging chads in the tree without increasing the parsimony score

    @param tree: the tree that we are supposed to find hanging chads in
    @return List[HangingChad]
    """
    hanging_chads = []
    for node in tree.traverse("preorder"):
        if node.is_root() or node.node_id is None:
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
        # TODO: spaggheti code
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

    logging.info("total number of hanging chads %d", len(hanging_chads))
    for c in hanging_chads:
        logging.info("Chad %s", c)
    return hanging_chads

def _prepare_nochad_fit_params(
        fit_params: Dict):
    if 'branch_len_inners' not in fit_params:
        return

    fit_params['target_lams'] = get_init_target_lams(fit_params['target_lams'].size)
    fit_params.pop('branch_len_inners', None)
    fit_params.pop('branch_len_offsets_proportion', None)

def _add_hanging_subtree(
        tree: CellLineageTree,
        hanging_subtree: CellLineageTree,
        chad_par_node_id: int):
    """
    Add `hanging_subtree` to the tree whereever we can find the node with orig_node_id = `chad_par_node_id`
    @return CellLineageTree with the new hanging chad
    """
    tree_copy = tree.copy()
    for node in tree_copy.traverse():
        if node.orig_node_id is not None and node.orig_node_id == chad_par_node_id:
            if node.is_leaf():
                new_child1 = node.copy()
                new_child1.add_features(
                    orig_node_id=None,
                    node_id=None,
                    nochad_id=None)
                new_child2 = new_child1.copy()
                node.add_child(new_child1)
                print("has spine children")
                # TODO: this attribute is really jenky -- use some other method to indicate
                node.add_feature('old_leaf', True)
                new_child1.add_child(new_child2)
                node = new_child1

            new_hanging_chad = hanging_subtree.copy()
            new_hanging_chad.up = None
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
    return tree_copy, num_nodes

def _make_nochad_tree(tree: CellLineageTree, hanging_chad: HangingChad):
    """
    @return CellLineageTree where we remove the hanging chad.
            Tree nodes are all relabeled with new node_id.
            In addition, we add some annotations to the no chad tree.
            attribute `orig_node_id`: the original node_id in the original tree
            attribute `nochad_id`: the node id in the nochad tree
    """
    nochad_tree = tree.copy()
    current_parent_id = None
    for node in nochad_tree.traverse("preorder"):
        node.add_feature("orig_node_id", node.node_id)
        if node.node_id == hanging_chad.node.node_id:
            current_parent_id = node.up.orig_node_id
            logging.info("current parent of chad %s", node.up.allele_events_list_str)
            node.detach()
    nochad_tree.label_node_ids()
    assert current_parent_id is not None

    for node in nochad_tree.traverse():
        # TODO can we use another way to mark the nodes? this is kinda jenky
        node.add_feature("nochad_id", node.node_id)
        # Track which nodes were originally leaves in the nochad tree
        # These nodes have nonsense values in the branch_len_inner param,
        # so we will make sure to ignore those values.
        node.add_feature("nochad_is_leaf", node.is_leaf())
        # Only trees with multifurcs have meaningful offset values...
        node.add_feature("nochad_is_multifurc", len(node.get_children()) > 2)

    return nochad_tree, current_parent_id

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

    _prepare_nochad_fit_params(fit_params)

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

def _create_random_list_possible_parents(
        hanging_chad: HangingChad,
        current_parent_id: int,
        max_chad_tune_search: int):
    """
    @param current_parent_id: current node id of the parent node
    @return list of randomly chosen possible parent nodes,
            the first element in the list is the current parent node of the hanging chad
    """
    other_chad_parents = [
            p for p in hanging_chad.possible_parents
            if p.node_id != current_parent_id]
    num_other_parents = len(other_chad_parents)
    random_order = np.random.choice(num_other_parents, num_other_parents, replace=False)
    chosen_idxs = random_order[:max_chad_tune_search - 1]
    possible_chad_parents = [
            p for p in hanging_chad.possible_parents
            if p.node_id == current_parent_id] + [other_chad_parents[idx] for idx in chosen_idxs]
    logging.info(
            "chad parent idxs considered %s (out of %d) (plus the orig one)",
            chosen_idxs,
            num_other_parents)
    return possible_chad_parents

def _create_warm_start_fit_params(
            no_chad_res: LikelihoodScorerResult,
            new_chad_tree: CellLineageTree,
            num_nodes: int):
    """
    Assemble the `fit_param` and `KnownModelParam` for fitting the tree with the hanging chad
    This does a warm start -- it copies over all model parameters and branch lengths
    @return Dict, KnownModelParam
    """
    fit_params = no_chad_res.get_fit_params()
    prev_branch_inners = fit_params["branch_len_inners"].copy()
    prev_branch_proportions = fit_params["branch_len_offsets_proportion"].copy()

    fit_params['dist_to_half_pen_param'] = 0
    branch_len_inners_mask = np.zeros(num_nodes, dtype=bool)
    branch_len_offsets_proportion_mask = np.zeros(num_nodes, dtype=bool)
    fit_params["branch_len_inners"] = np.ones(num_nodes) * 1e-10
    fit_params["branch_len_offsets_proportion"] = np.ones(num_nodes) * 0.45 + np.random.rand(num_nodes) * 0.1
    for node in new_chad_tree.traverse():
        if node.nochad_id is not None:
            branch_len_inners_mask[node.node_id] = True
            branch_len_offsets_proportion_mask[node.node_id] = True

            # Copy over existing branch length estimates -- it it matches an existing branch in the no-chad tree
            nochad_id = int(node.nochad_id)
            node_id = int(node.node_id)
            if not node.nochad_is_leaf:
                # Only copy branch_len_inners for non-leaf nodes (from the nochad tree) because branch_len_inner values
                # are meaningless for leaf nodes
                fit_params["branch_len_inners"][node_id] = max(
                    prev_branch_inners[nochad_id] - 1e-9, 1e-10)

            if not node.is_root() and node.up.nochad_is_multifurc:
                fit_params["branch_len_offsets_proportion"][node_id] = prev_branch_proportions[nochad_id]
            else:
                # If this wasnt a multifurc and has suddenly become one,
                # we will just force it to be a true resolved multifurcation.
                fit_params["branch_len_offsets_proportion"][node_id] = 1e-10

            if hasattr(node, 'old_leaf'):
                print("spine assignments...")
                node_up_dist_to_root = no_chad_res.train_history[-1]['dist_to_roots'][node.up.nochad_id]
                node_dist_to_root = no_chad_res.train_history[-1]['dist_to_roots'][node.nochad_id]
                node_branch_inner = node_dist_to_root - node_up_dist_to_root
                if node.up.nochad_is_multifurc:
                    fit_params["branch_len_inners"][node_id] = node_branch_inner * prev_branch_proportions[nochad_id]
                    fit_params["branch_len_offsets_proportion"][node_id] = 1 - 1e-10
                else:
                    fit_params["branch_len_inners"][node_id] = 1e-10
                    fit_params["branch_len_offsets_proportion"][node_id] = 1e-10

                child = node.get_children()[0]
                fit_params["branch_len_inners"][child.node_id] = node_branch_inner * 0.5 * (1 - prev_branch_proportions[nochad_id])
                print("unkonwn", [c.node_id for c in node.get_descendants()])
    print(fit_params["branch_len_inners"][np.logical_not(branch_len_inners_mask)])
    print("where unkonwn", np.where(np.logical_not(branch_len_inners_mask)))

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
    assert len(hanging_chad.possible_parents) > 1

    # Remove my hanging chad from the orig tree
    # Also track the parent we took it off from so that
    # we always have an option of NOT moving the hanging chad
    nochad_tree, current_parent_id = _make_nochad_tree(tree, hanging_chad)
    # Fit the nochad tree
    no_chad_res = _fit_nochad_result(
        nochad_tree,
        bcode_meta,
        args,
        fit_params,
        assessor=assessor)

    # For all possible places to hang the hanging chad,
    # create a likelihood worker so that we can evaluate how good the
    # log likelihood would be.
    possible_chad_parents = _create_random_list_possible_parents(
            hanging_chad,
            current_parent_id,
            args.max_chad_tune_search)
    worker_list = []
    full_chad_trees = []
    for parent_idx, chad_par in enumerate(possible_chad_parents):
        print("parent idx", parent_idx)

        # assemble the full tree to track the candidate tree
        full_tree, _ = _add_hanging_subtree(
                no_chad_res.orig_tree,
                hanging_chad.node,
                chad_par.node_id)
        full_chad_trees.append(full_tree)

        # Pick a random leaf from the hanging chad -- do not use the entire hanging chad
        # This is because the entire hanging chad might have multiple leaves and their
        # branch length assignment is ambigious.
        new_chad_tree, num_nodes_new_chad_tree = _add_hanging_subtree(
                no_chad_res.orig_tree,
                hanging_chad.get_rand_leaf_node(),
                chad_par.node_id)
        warm_start_fit_params, warm_start_known_params = _create_warm_start_fit_params(
            no_chad_res,
            new_chad_tree,
            num_nodes_new_chad_tree)

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
    worker_results = [w[0] for w in job_manager.run()]

    # Aggregate the results
    chad_tune_res = _create_chad_results(
        worker_results,
        possible_chad_parents,
        full_chad_trees,
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
        fit_results: List[List[LikelihoodScorerResult]],
        chad_parent_candidates: List[CellLineageTree],
        full_chad_trees: List[CellLineageTree],
        no_chad_res: LikelihoodScorerResult,
        hanging_chad: HangingChad,
        scratch_dir: str):
    """
    Process the results and aggregate them in a HangingChadTuneResult

    @return HangingChadTuneResult
    """
    assert len(fit_results) == len(chad_parent_candidates)
    assert len(fit_results) == len(full_chad_trees)

    new_chad_results = [
        HangingChadResult(
            fit_res[0].log_lik[0],
            hanging_chad.node,
            chad_par,
            full_chad_tree,
            fit_res)
        for fit_res, chad_par, full_chad_tree in zip(fit_results, chad_parent_candidates, full_chad_trees)]
    return HangingChadTuneResult(no_chad_res, new_chad_results)
