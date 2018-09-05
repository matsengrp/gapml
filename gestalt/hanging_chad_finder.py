import numpy as np
from typing import List, Dict
import logging

from allele_events import Event
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import SubprocessManager
from split_data import create_kfold_trees, create_kfold_barcode_trees, TreeDataSplit
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from common import get_randint

"""
Hanging chad is our affectionate name for the inter-target cuts that have ambiguous placement
in the tree. For example, if we don't know if the 1-3 inter-target cut is a child of the root
node or a child of another allele with a 2-2 intra-target cut.
"""

class HangingChadResult:
    def __init__(self,
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
    def __init__(self,
            no_chad_res: LikelihoodScorerResult,
            new_chad_results: List[HangingChadResult]):
        self.no_chad_res = no_chad_res
        self.new_chad_results = new_chad_results

    def get_best_chad(self):
        raise NotImplementedError()


class HangingChad:
    def __init__(self,
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
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        init_model_params,
        max_num_chad_parents: int = 20):
    hanging_chad_dict = get_chads(tree)
    sorted_chad_keys = sorted(list(hanging_chad_dict.keys()))
    logging.info("total number of hanging chads %d", len(sorted_chad_keys))
    logging.info(sorted_chad_keys)

    for idx, chad_evt in enumerate(sorted_chad_keys):
        hanging_chad = hanging_chad_dict[chad_evt]
        logging.info("chad %d: %s, num possible parents %d",
                idx,
                chad_evt,
                len(hanging_chad.possible_parents))

    # TODO: change this to do something other than a fixed chad
    for chad_evt in sorted_chad_keys[3:]:
        hanging_chad = hanging_chad_dict[chad_evt]
        logging.info("chad: %s, num possible parents %d", chad_evt, len(hanging_chad.possible_parents))
        if len(hanging_chad.possible_parents) == 1:
            continue

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
            init_model_param_list = [init_model_params],
            known_params = args.known_params,
            abundance_weight = args.abundance_weight).run_worker(None)[0]

        warm_start_params = no_chad_res.get_fit_params()
        prev_branch_inners = warm_start_params["branch_len_inners"].copy()
        prev_branch_proportions = warm_start_params["branch_len_offsets_proportion"].copy()

        worker_list = []
        possible_chad_parents = hanging_chad.possible_parents[:max_num_chad_parents]
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
            print(tree_copy.get_ascii(attributes=["node_id"]))
            print("attached....", chad_evt, chad_par.allele_events_list_str)
            print("with new chad tree leaves", len(tree_copy))

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
            print(warm_start_params["branch_len_inners"])
            print(warm_start_params["branch_len_offsets_proportion"])

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
                init_model_param_list = [warm_start_params],
                known_params = args.known_params,
                abundance_weight = args.abundance_weight)
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

        best_hanging_chad_idx = np.argmax([
                chad_res.score for chad_res in chad_results])
        for chad_res in chad_results:
            logging.info("Chad res: %s", str(chad_res))

        #TODO: remove this. we are only trying one hanging chad first
        break
    # TODO: right now we just flatten the list
    return chad_results, no_chad_res

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
