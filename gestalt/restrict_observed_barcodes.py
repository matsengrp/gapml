"""
Apply filtration -- we can only observe the first `n` alleles in each cell.
"""
from __future__ import division, print_function
import os
import sys
import argparse
import logging
import six
import itertools
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

import collapsed_tree as collapser
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
from likelihood_scorer import LikelihoodScorer
from clt_observer import ObservedAlignedSeq
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import BatchSubmissionManager
from plot_mrca_matrices import plot_mrca_matrix
from common import save_data
from common import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description='Collapse data based on first n alleles')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--model-file',
        type=str,
        help='pkl file with true model (optional)')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/restrict_log.txt",
        help='pkl file with true model')
    parser.add_argument(
        '--num-barcodes',
        type=int,
        default=1,
        help="Number of the barcodes we actually observe")
    parser.add_argument(
        '--out-obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='name of the output pkl file with collapsed observed sequence data')
    parser.add_argument(
        '--out-collapsed-tree-file',
        type=str,
        default="_output/collapsed_tree_b1.pkl",
        help='name of the output pkl file with collapsed tree')
    parser.add_argument(
        '--submit-srun',
        action='store_true',
        help='is using slurm to submit jobs')

    args = parser.parse_args()
    args.scratch_dir = "_output/scratch"
    create_directory(args.out_obs_file)
    return args

def collapse_obs_leaves_by_first_alleles(
        obs_leaves: List[ObservedAlignedSeq],
        num_barcodes: int):
    """
    Collapse the observed data based on the first `num_barcodes` alleles
    @return List[ObservedAlignedSeq]
    """
    obs_dict = {}
    for obs in obs_leaves:
        if obs.allele_list is not None:
            obs.set_allele_list(obs.allele_list.create_truncated_version(num_barcodes))
        else:
            obs.allele_events_list = obs.allele_events_list[:num_barcodes]

        # Make sure to keep unique observations and update abundance accordingly
        obs_key = str(obs)
        if obs_key in obs_dict:
            obs_dict[obs_key].abundance += obs.abundance
        else:
            obs_dict[obs_key] = obs

    return list(obs_dict.values())

def get_duplicate_indpt_alleles(coll_tree: CellLineageTree):
    """
    @param coll_tree: a collapsed cell lineage tree with potentially duplicate alleles in the leaves
                        these are independently-arisen alleles
    @return List[List[Tuple]], where this is a list grouping independently-arisen leaves if
                                they have the same allele
                            Tuple is (idx in this list, leaf node, leaf parent)
    """
    # Map each allele to the leaves with the same allele
    uniq_alleles = dict()
    for leaf in coll_tree:
        if leaf.allele_events_list_str in uniq_alleles:
            uniq_alleles[leaf.allele_events_list_str].append(leaf)
        else:
            uniq_alleles[leaf.allele_events_list_str] = [leaf]

    # Find the alleles that map to more than one leaf
    duplicate_indpt_alleles = []
    for key, leaves in uniq_alleles.items():
        if len(leaves) > 1:
            duplicate_indpt_alleles.append([
                (i, leaf, leaf.up) for i, leaf in enumerate(leaves)])

    for duplicate_grp in duplicate_indpt_alleles:
        logging.info("duplicate group %s", duplicate_grp[0][1].allele_events_list_str)

    return duplicate_indpt_alleles

def _detach_leaves(leaves_kept: Tuple[Tuple], duplicate_indpt_alleles: List[List[Tuple]]):
    """
    @param leaves_kept: Represents which leaf we keep from each of the groups of duplicate alleles
                        The outer tuple corresponds to each of the groups in `duplicate_indpt_alleles`.
                        The inner tuple is the one we chose in each of the groups
    @param duplicate_indpt_alleles: List[List[Tuple]], where this is a list grouping independently-arisen leaves
                        (see get_duplicate_indpt_alleles)

    Removes the leaves that are not in `leaves_kept`
    """
    for group_same_alleles_idx, (leaf_kept_idx, leaf_kept, _) in enumerate(leaves_kept):
        for leaf_idx, other_leaf, _ in duplicate_indpt_alleles[group_same_alleles_idx]:
            if leaf_kept_idx != leaf_idx:
                other_leaf.detach()

def _reattach_leaves(leaves_kept: Tuple[Tuple], duplicate_indpt_alleles: List[List[Tuple]]):
    """
    Undo what we did in `_detach_leaves`
    """
    for group_same_alleles_idx, (leaf_kept_idx, leaf_kept, _) in enumerate(leaves_kept):
        for leaf_idx, other_leaf, other_leaf_parent in duplicate_indpt_alleles[group_same_alleles_idx]:
            if leaf_kept_idx != leaf_idx:
                other_leaf_parent.add_child(other_leaf)

def remove_same_parent_duplicates(duplicate_indpt_alleles: List[List[Tuple]]):
    """
    @param duplicate_indpt_alleles: List of independently-arisen leaves, grouped by those with the same allele
                                    Tuple is (idx in this list, leaf node, leaf parent)

    This will detach duplicate allele leaves that have the same parent (detach all but one)
    In addition....
    @return a sublist of `duplicate_indpt_alleles` where those leaves do not have the same parent node
    """
    filtered_duplicate_indpt_alleles = []
    for idx, duplicate_grp in enumerate(duplicate_indpt_alleles):
        # Keep only the leaves that have different parent nodes
        new_duplicate_grp = [duplicate_grp[0]]
        for leaf_idx, leaf, leaf_parent in duplicate_grp[1:]:
            for _, _, prev_parent in new_duplicate_grp:
                if leaf_parent == prev_parent:
                    leaf.detach()
            else:
                new_duplicate_grp.append((leaf_idx, leaf, leaf_parent))

        if len(new_duplicate_grp) == 1:
            logging.info("Dropping duplicate group %d since all same parent", idx)
        else:
            logging.info("Not the same parent for duplicate group %d", idx)
            filtered_duplicate_indpt_alleles.append(new_duplicate_grp)
    return filtered_duplicate_indpt_alleles

def make_likelihood_scorer(tree: CellLineageTree, true_model_dict: Dict, name: str):
    """
    @param tree: make a likelihood scorer for this tree, also use this tree as source for
                true branch lengths
    @param true_model_dict: use this to set the init_model_params in the LikelihoodScorer
    @param name: name for the likelihood scorer (just for debugging)

    @return LikelihoodScorer with model params set according to `true_model_dict`
                and branch lengths according to that in `tree`
    """
    collapser._remove_single_child_unobs_nodes(tree)
    tree.label_node_ids()
    num_nodes = tree.get_num_nodes()
    bcode_meta = true_model_dict["bcode_meta"]

    # Plug in the true branch lengths into the true model param dictionary
    # (first make a copy)
    param_dict = {key: val for key, val in true_model_dict["true_model_params"].items()}
    br_lens = np.zeros(num_nodes)
    for node in tree.traverse():
        br_lens[node.node_id] = node.dist
        node.resolved_multifurcation = True
    param_dict["branch_len_inners"] = br_lens
    param_dict["branch_len_offsets_proportion"] = np.zeros(num_nodes)
    param_dict.pop('cell_type_lams', None)

    scorer = LikelihoodScorer(
        0, # seed
        tree,
        true_model_dict["bcode_meta"],
        log_barr = 0, # no penalty
        target_lam_pen = 0, # no penalty
        max_iters = 0,
        num_inits = 3,
        transition_wrap_maker = TransitionWrapperMaker(tree, bcode_meta),
        tot_time = true_model_dict["time"],
        init_model_params = param_dict)
    return scorer

def get_highest_likelihood_single_appearance_tree(
        coll_tree: CellLineageTree,
        true_model_dict: Dict,
        scratch_dir: str,
        submit_srun: bool):
    """
    @param tree: a collapsed tree where each allele may appear more than once because they arose independently in the tree
    @param true_model_dict: a Dict containing all the parameters needed to instantiate the CLTLikelihoodModel

    To get the likelihoods of the different collapsed trees, we submit jobs to slurm

    @return CellLineageTree where each allele appears once in the leaves.
            We pick the tree by finding the subtree with the highest likelihood according to the true model params
    """
    # Get the duplicate independent alleles
    duplicate_indpt_alleles = get_duplicate_indpt_alleles(coll_tree)
    num_indpt_groups = len(duplicate_indpt_alleles)
    logging.info("There are %d groups of independently arisen, identical alleles", num_indpt_groups)
    if num_indpt_groups == 0:
        # No duplicates. Yay!
        return coll_tree

    # Do a first pass fix -- if all the duplicate independent alleles have the same parents, we can just keep one
    # and this problem is easily solved.
    filtered_duplicate_indpt_alleles = remove_same_parent_duplicates(duplicate_indpt_alleles)
    if len(filtered_duplicate_indpt_alleles) == 0:
        # No more duplicates. Yay!
        return coll_tree

    # There are duplicates. So let's consider all ways to drop duplicates
    leaves_kept_combos = list(itertools.product(*filtered_duplicate_indpt_alleles))
    logging.info("There are a total of %d trees to compare", len(leaves_kept_combos))

    scorers = []
    for keep_idx, leaves_kept in enumerate(leaves_kept_combos):
        # Detach the designated leaves from the collapsed tree
        _detach_leaves(leaves_kept, filtered_duplicate_indpt_alleles)

        scorer = make_likelihood_scorer(
                coll_tree.copy("cpickle"),
                true_model_dict,
                "likelihood scorer %d" % keep_idx)
        scorers.append(scorer)

        # Undo our changes to the collapsed tree
        _reattach_leaves(leaves_kept, filtered_duplicate_indpt_alleles)

    # Submit jobs to slurm
    if submit_srun:
        job_manager = BatchSubmissionManager(scorers, None, len(leaves_kept_combos), scratch_dir)
        scorer_results = job_manager.run(successful_only=True)
        assert len(scorer_results) == len(leaves_kept_combos)
    else:
        logging.info("Running locally")
        scorer_results = [(w.run_worker(None), w) for w in scorers]
    logging.info([res.pen_log_lik for (res, _) in scorer_results])

    # Get the one with the highest log likelihood
    best_leaves_kept = None
    best_log_lik = -np.inf
    for (res, scorer) in scorer_results:
        if res.pen_log_lik > best_log_lik:
            best_coll_tree = scorer.tree
            best_log_lik = res.pen_log_lik

    return best_coll_tree

def collapse_tree_by_first_alleles(true_model_dict: Dict, num_barcodes: int):
    """
    Generate the collapsed tree based on first `num_barcodes` alleles
    @return CellLineageTree
    """
    # Truncate the number of alleles observed at the internal nodes
    true_subtree = true_model_dict["true_subtree"]
    for node in true_subtree.traverse():
        node.set_allele_list(node.allele_list.create_truncated_version(num_barcodes))
    # Create the collapsed CLT according to the first `args.num_barcodes` alleles
    collapsed_subtree = collapser.collapse_ultrametric(true_subtree)
    return collapsed_subtree

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    bcode_meta = obs_data_dict["bcode_meta"]
    orig_num_barcodes = bcode_meta.num_barcodes
    assert args.num_barcodes <= orig_num_barcodes

    # Update the barcode metadata to have the collapsed number of barcodes
    bcode_meta.num_barcodes = args.num_barcodes

    # Now start collapsing the observations by first n alleles
    raw_obs_leaves = obs_data_dict["obs_leaves"]
    obs_data_dict["obs_leaves"] = collapse_obs_leaves_by_first_alleles(
            raw_obs_leaves,
            args.num_barcodes)
    logging.info(
        "Number of uniq obs after restricting to first %d alleles: %d",
        args.num_barcodes,
        len(obs_data_dict["obs_leaves"]))
    save_data(obs_data_dict, args.out_obs_file)

    # Generate the true collapsed tree
    if args.model_file is not None:
        with open(args.model_file, "rb") as f:
            true_model_dict = six.moves.cPickle.load(f)
        collapsed_clt = collapse_tree_by_first_alleles(
                true_model_dict,
                args.num_barcodes)
        logging.info(collapsed_clt.get_ascii(
            attributes=["allele_events_list_str"],
            show_internal=True))

        selected_collapsed_clt = get_highest_likelihood_single_appearance_tree(
            collapsed_clt,
            true_model_dict,
            args.scratch_dir,
            args.submit_srun)

        # Assert no duplicate alleles in the collapsed tree
        assert len(selected_collapsed_clt) == len(obs_data_dict["obs_leaves"])
        if args.num_barcodes == orig_num_barcodes:
            assert len(obs_data_dict["obs_leaves"]) == len(raw_obs_leaves)

        save_data(selected_collapsed_clt, args.out_collapsed_tree_file)

        # Plot the MRCA matrix of the true collapsed tree for fun
        out_png = args.out_collapsed_tree_file.replace(".pkl", "_mrca.png")
        plot_mrca_matrix(selected_collapsed_clt, out_png)

if __name__ == "__main__":
    main()
