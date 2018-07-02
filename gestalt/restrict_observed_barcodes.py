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
from typing import Dict, List

import collapsed_tree
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
from clt_observer import ObservedAlignedSeq
from transition_wrapper_maker import TransitionWrapperMaker
from parallel_worker import BatchSubmissionManager
from plot_mrca_matrices import plot_mrca_matrix
from common import save_data

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
        default="_output/true_model.pkl",
        help='pkl file with true model')
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

    args = parser.parse_args()
    args.scratch_dir = "_output/scratch"
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
        obs.set_allele_list(obs.allele_list.create_truncated_version(num_barcodes))

        # Make sure to keep unique observations and update abundance accordingly
        obs_key = str(obs)
        if obs_key in obs_dict:
            obs_dict[obs_key].abundance += obs.abundance
        else:
            obs_dict[obs_key] = obs

    return list(obs_dict.values())

def get_log_likelihood(true_topology: CellLineageTree, true_model_dict: Dict):
    """
    @param true_topology: CellLineageTree that we are assuming to have the true topology/branch lengths
    @param true_model_dict: Dict containing all the info needed for creating the true model
                        same Dict as the one in `generate_data.py`

    @return log likelihood of `true_topology` for the given model params,
            using the branch lengths in `true_topology` as the truth
    """
    with tf.Session() as sess:
        # Prepare the tree topology
        true_topology.label_node_ids()
        num_nodes = true_topology.get_num_nodes()

        # Create the model
        clt_model = CLTLikelihoodModel(
            true_topology,
            true_model_dict["bcode_meta"],
            sess,
            true_model_dict["true_model_params"]["target_lams"])
        tf.global_variables_initializer().run()

        # Create the param dict, with the true branch lengths
        param_dict = true_model_dict["true_model_params"]
        br_lens = np.zeros(num_nodes)
        for node in true_topology.traverse():
            br_lens[node.node_id] = node.dist
        param_dict["branch_len_inners"] = br_lens
        param_dict["branch_len_offsets_proportion"] = np.zeros(num_nodes)

        # Create the log likelihood node -- this requires getting the transition matrix wrappers
        transition_wrap_maker = TransitionWrapperMaker(
            true_topology,
            true_model_dict["bcode_meta"])
        transition_wrappers = transition_wrap_maker.create_transition_wrappers()
        logging.info("Done creating transition wrappers")
        clt_model.create_log_lik(transition_wrappers, create_gradient=False)

        # Update the model to have the true model params
        clt_model.set_params_from_dict(param_dict)

        # Finally get the log likelihood of the data
        log_lik, _ = clt_model.get_log_lik()

    return log_lik

def get_highest_likelihood_single_appearance_tree(coll_tree: CellLineageTree, true_model_dict: Dict, args):
    """
    @param tree: a collapsed tree where each allele may appear more than once because they arose independently in the tree
    @param true_model_dict: a Dict containing all the parameters needed to instantiate the CLTLikelihoodModel

    @return CellLineageTree where each allele appears once in the leaves.
            We pick the tree by finding the subtree with the highest likelihood according to the true model params
    """
    def _detach_leaves(leaves_kept, duplicate_indpt_alleles):
        # Removes the leaves that are not in `leaves_kept`
        for group_same_alleles_idx, (leaf_kept_idx, leaf_kept, _) in enumerate(leaves_kept):
            for leaf_idx, other_leaf, _ in duplicate_indpt_alleles[group_same_alleles_idx]:
                if leaf_kept_idx != leaf_idx:
                    other_leaf.detach()

    def _reattach_leaves(leaves_kept, duplicate_indpt_alleles):
        # Undo what we did in `_detach_leaves`
        for group_same_alleles_idx, (leaf_kept_idx, leaf_kept, _) in enumerate(leaves_kept):
            for leaf_idx, other_leaf, other_leaf_parent in duplicate_indpt_alleles[group_same_alleles_idx]:
                if leaf_kept_idx != leaf_idx:
                    other_leaf_parent.add_child(other_leaf)

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

    num_indpt_groups = len(duplicate_indpt_alleles)
    logging.info("There are %d groups of independently arisen, identical alleles", num_indpt_groups)
    if num_indpt_groups == 0:
        # No duplicates. Yay!
        return coll_tree

    # There are duplicates. So let's consider all ways to drop duplicates
    leaves_kept_combos = list(itertools.product(*duplicate_indpt_alleles))

    #best_leaves_kept = leaves_kept_combos[0]
    scorers = []
    for keep_idx, leaves_kept in enumerate(leaves_kept_combos):
        # Detach the designated leaves
        _detach_leaves(leaves_kept, duplicate_indpt_alleles)

        bcode_meta = true_model_dict["bcode_meta"]
        new_tree = coll_tree.copy()
        scorer = LikelihoodScorer(
            0,
            new_tree,
            true_model_dict["bcode_meta"],
            log_barr = 0,
            max_iters = 0,
            transition_wrap_maker = TransitionWrapperMaker(new_tree, bcode_meta),
            init_model_params = true_model_dict["true_model_params"])
        scorers.append(scorer)

        # Undo our changes to the tree
        _reattach_leaves(leaves_kept, duplicate_indpt_alleles)

    job_manager = BatchSubmissionManager(scorers, None, len(leaves_kept_combos), args.scratch_dir)
    scorer_results = job_manager.run(successful_only=True)

    best_leaves_kept = None
    best_log_lik = -np.inf
    for (res, scorer) in scorer_results:
        if res.train_history[-1].pen_log_lik > best_log_lik:
            best_coll_tree = scorer.tree
            best_log_lik = res.train_history[-1].pen_log_lik

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
    collapsed_subtree = collapsed_tree.collapse_ultrametric(true_subtree)
    return collapsed_subtree

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    bcode_meta = obs_data_dict["bcode_meta"]
    assert args.num_barcodes <= bcode_meta.num_barcodes

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
    with open(args.model_file, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)
    collapsed_clt = collapse_tree_by_first_alleles(
            true_model_dict,
            args.num_barcodes)

    selected_collapsed_clt = get_highest_likelihood_single_appearance_tree(
        collapsed_clt,
        true_model_dict,
        args)

    # Assert no duplicate alleles in the collapsed tree
    assert len(selected_collapsed_clt) == len(obs_data_dict["obs_leaves"])
    if args.num_barcodes == bcode_meta.num_barcodes:
        assert len(obs_data_dict["obs_leaves"]) == len(raw_obs_leaves)

    save_data(selected_collapsed_clt, args.out_collapsed_tree_file)

    # Plot the MRCA matrix of the true collapsed tree for fun
    out_png = args.out_collapsed_tree_file.replace(".pkl", "_mrca.png")
    plot_mrca_matrix(selected_collapsed_clt, out_png)

if __name__ == "__main__":
    main()
