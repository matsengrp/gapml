import numpy as np
from typing import List, Dict
import logging

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import SubprocessManager
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from common import get_randint
from tree_distance import TreeDistanceMeasurerAgg

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
            fit_res: LikelihoodScorerResult):
        self.score = score
        self.chad_node = chad_node
        self.parent_node = parent_node
        self.fit_res = fit_res

    def __str__(self):
        return "%s=>%s, score=%f" % (
                self.parent_node.allele_events_list_str,
                self.chad_node.allele_events_list_str,
                self.score)


class HangingChadTuneResult:
    def __init__(
            self,
            no_chad_res: LikelihoodScorerResult,
            new_chad_results: List[HangingChadResult]):
        self.no_chad_res = no_chad_res
        self.new_chad_results = new_chad_results

    def get_best_result(self):
        best_chad_idx = np.argmax([
                chad_res.score for chad_res in self.new_chad_results])
        best_fit_res = self.new_chad_results[best_chad_idx].fit_res

        fit_params = best_fit_res.get_fit_params()
        fit_params.pop('branch_len_inners', None)
        fit_params.pop('branch_len_offsets_proportion', None)
        return best_fit_res.orig_tree, fit_params, best_fit_res


class HangingChad:
    def __init__(
            self,
            node: CellLineageTree,
            possible_parents: List[CellLineageTree],
            parsimony_contribution: int):
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
        curr_par_event_str = node.up.allele_events_list_str

        # Determine if there is another place in the tree where we can preserve
        # parsimony score
        # TODO: this still does not find every possible location for the hanging chads
        # it is possible that there is a node omitted in the parsimony tree that
        # this node can be a child of
        possible_parents = [node.up]
        for potential_par_node in tree.get_descendants():
            if potential_par_node.allele_events_list_str == curr_par_event_str:
                continue
            potential_score = _get_parsimony_score(node, potential_par_node)
            if potential_score is None or parsimony_score == 0:
                continue

            if potential_score == parsimony_score:
                possible_parents.append(potential_par_node)
        if len(possible_parents) > 1:
            hanging_chads.append(HangingChad(
                    node,
                    possible_parents,
                    parsimony_score))

    logging.info("Number of hanging chads found: %d", len(hanging_chads))
    return hanging_chads


def tune(
        hanging_chad: HangingChad,
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        fit_params: Dict,
        dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    Tune the given hanging chad
    @return HangingChadTuneResult
    """
    assert len(hanging_chad.possible_parents) > 1

    # Remove my hanging chad from the orig tree
    nochad_tree = tree.copy()
    for node in nochad_tree.traverse():
        node.add_feature("orig_node_id", node.node_id)
        if node.node_id == hanging_chad.node.node_id:
            node.detach()
    nochad_tree.label_node_ids()
    print("no chad tree leaves", len(nochad_tree))

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
        known_params=args.known_params).run_worker(None)[0]

    warm_start_params = no_chad_res.get_fit_params()
    prev_branch_inners = warm_start_params["branch_len_inners"].copy()
    prev_branch_proportions = warm_start_params["branch_len_offsets_proportion"].copy()

    worker_list = []
    possible_chad_parents = hanging_chad.possible_parents[:args.max_chad_tune_search]
    for chad_par in possible_chad_parents:
        # From the no chad tree, add back the hanging chad to the designated parent
        tree_copy = nochad_tree.copy()
        for node in tree_copy.traverse():
            if not node.is_leaf():
                # only copy over branch length inners/proportions if not leaf
                node.add_feature("nochad_id", node.node_id)
            else:
                node.add_feature("nochad_id", None)

        for node in tree_copy.traverse():
            if node.orig_node_id is not None and node.orig_node_id == chad_par.node_id:
                if node.is_leaf():
                    new_child = node.copy()
                    new_child.add_feature("orig_node_id", None)
                    new_child.add_feature("node_id", None)
                    new_child.add_feature("nochad_id", None)
                    node.add_child(new_child)
                new_hanging_chad = hanging_chad.node.copy()
                new_hanging_chad.add_feature("orig_node_id", None)
                new_hanging_chad.add_feature("node_id", None)
                new_hanging_chad.add_feature("nochad_id", None)
                for descendant in new_hanging_chad.get_descendants():
                    descendant.add_feature("orig_node_id", None)
                    descendant.add_feature("node_id", None)
                    descendant.add_feature("nochad_id", None)
                node.add_child(new_hanging_chad)

        num_nodes = tree_copy.label_node_ids()

        # warm start the branch length estimates
        # TODO: doesn't transfer over every single branch length estimate... is that ok?
        # Is there a way to transfer over more branch length estimates?
        warm_start_params = no_chad_res.get_fit_params()
        warm_start_params["branch_len_inners"] = np.ones(num_nodes) * 1e-10
        warm_start_params["branch_len_offsets_proportion"] = np.ones(num_nodes) * 1e-10
        for node in tree_copy.traverse():
            if node.nochad_id is not None:
                nochad_id = int(node.nochad_id)
                node_id = int(node.node_id)
                warm_start_params["branch_len_inners"][node_id] = max(
                        prev_branch_inners[nochad_id] - 1e-9, 1e-10)
                warm_start_params["branch_len_offsets_proportion"][node_id] = prev_branch_proportions[nochad_id]

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
            dist_measurers=dist_measurers)
        worker_list.append(worker)

    if args.num_processes > 1 and len(worker_list) > 1:
        job_manager = SubprocessManager(
                worker_list,
                None,
                args.scratch_dir,
                threads=args.num_processes)
        worker_results = job_manager.run()
    else:
        worker_results = [(w.run_worker(None), w) for w in worker_list]

    chad_results = [
        create_chad_result(worker_res[0][0], no_chad_res, hanging_chad, chad_par)
        for worker_res, chad_par in zip(worker_results, possible_chad_parents)]

    for chad_res in chad_results:
        logging.info("Chad res: %s", str(chad_res))

    return HangingChadTuneResult(no_chad_res, chad_results)


def create_chad_result(
        new_chad_res: LikelihoodScorerResult,
        no_chad_res: LikelihoodScorerResult,
        hanging_chad: HangingChad,
        chad_par: CellLineageTree):
    """
    @return HangingChadResult
    """
    new_chad_targs = new_chad_res.get_all_target_params()
    no_chad_targs = no_chad_res.get_all_target_params()
    stability_score = -np.linalg.norm(new_chad_targs - no_chad_targs)
    return HangingChadResult(
        stability_score,
        hanging_chad.node,
        chad_par,
        new_chad_res)
