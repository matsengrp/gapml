"""
Get tree topologies from parsimony
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
import argparse
import time
import random
import logging
from pathlib import Path
import six

from cell_lineage_tree import CellLineageTree
from clt_observer import ObservedAlignedSeq
from barcode_metadata import BarcodeMetadata
from tree_distance import TreeDistanceMeasurer, UnrootRFDistanceMeasurer
from clt_estimator import CLTParsimonyEstimator
from collapsed_tree import collapse_zero_lens
from constants import *
from common import *

def parse_args():
    parser = argparse.ArgumentParser(description='fit topology and branch lengths for GESTALT')
    parser.add_argument(
        '--obs-data-pkl',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--true-model-pkl',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--out-folder',
        type=str,
        default="_output",
        help='folder to put output in')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument(
            '--seed',
            type=int,
            default=40,
            help="Random number generator seed")
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--num-jumbles',
            type=int,
            default=1)
    parser.add_argument('--max-random',
            type=int,
            default=1)
    parser.add_argument('--max-random-multifurc',
            type=int,
            default=1)
    parser.add_argument('--max-best',
            type=int,
            default=0)
    parser.add_argument('--max-best-multifurc',
            type=int,
            default=0)

    args = parser.parse_args()
    if args.max_best_multifurc or args.max_best:
        # Require having true tree to know what is a "best" tree
        assert args.true_model_pkl is not None

    args.log_file = "%s/parsimony_log.txt" % args.out_folder
    print("Log file", args.log_file)

    # check that there is no infile in the current folder -- this will
    # screw up mix because it will use the wrong input file
    my_file = Path("infile")
    assert not my_file.exists()

    args.scratch_dir = os.path.join(args.out_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args

def get_sorted_parsimony_trees(
        parsimony_trees: List[CellLineageTree],
        oracle_measurer: TreeDistanceMeasurer):
    """
    @return a sorted list of tree tuples (tree, tree dist) based on
            distance to the true tree
    """
    oracle_tuples = []
    for pars_tree in parsimony_trees:
        tree_dist = oracle_measurer.get_dist(pars_tree)
        oracle_tuples.append((
            pars_tree,
            tree_dist))

    oracle_tuples = list(sorted(oracle_tuples, key=lambda tup: tup[1]))
    parsimony_dists = [tup[1] for tup in oracle_tuples]

    logging.info("Uniq parsimony %s distances: %s", oracle_measurer.name, np.unique(parsimony_dists))
    logging.info("Mean parsimony %s distance: %f", oracle_measurer.name, np.mean(parsimony_dists))
    logging.info("Min parsimony %s distance: %d", oracle_measurer.name, np.min(parsimony_dists))

    return oracle_tuples

def get_bifurc_multifurc_trees(
        tree_list: List[CellLineageTree],
        max_bifurc: int,
        max_multifurc: int,
        selection_type: str,
        distance_cls,
        scratch_dir: str):
    """
    @param distance_cls: a TreeDistanceMeasurer class for finding unique collapsed trees

    @return a list of dictionaries with entries
                selection_type
                whether or not a collapsed tree
                idx -- uniq ID for this tree topology within selection type
                aux -- random other info
                tree -- the tree topology
            The list has no more than `max_bifurc` bifurcating trees and
            no more than `max_multifurc` collapsed/multifurcating trees.
    """
    trees_to_output = []
    tree_measurers = []
    for i, tree_tuple in enumerate(tree_list):
        num_multifurc = len(tree_measurers)
        if i > max_bifurc and num_multifurc > max_multifurc:
            break

        tree = tree_tuple[0]
        if i < max_bifurc:
            # Append to trees
            trees_to_output.append({
                'selection_type': selection_type,
                'multifurc': False,
                'idx': i,
                'aux': tree_tuple[1],
                'tree': tree})

        if num_multifurc < max_multifurc:
            # Check if this multifurc tree is unique
            has_match = False
            coll_tree = collapse_internally_labelled_tree(tree)
            for measurer in tree_measurers:
                dist = measurer.get_dist(coll_tree)
                has_match = dist == 0
                if has_match:
                    break

            if not has_match:
                tree_measurers.append(distance_cls(coll_tree, scratch_dir))
                # Append to trees
                trees_to_output.append({
                    'selection_type': selection_type,
                    'multifurc': True,
                    'idx': num_multifurc,
                    'aux': None,
                    'tree': coll_tree})
    return trees_to_output

def collapse_internally_labelled_tree(tree: CellLineageTree):
    coll_tree = tree.copy("deepcopy")
    for n in coll_tree.traverse():
        n.name = n.allele_events_list_str
        if not n.is_root():
            if n.allele_events_list_str == n.up.allele_events_list_str:
                n.dist = 0
            else:
                n.dist = 1
    coll_tree = collapse_zero_lens(coll_tree)
    return coll_tree

def get_parsimony_trees(
        obs_leaves: List[ObservedAlignedSeq],
        args,
        bcode_meta: BarcodeMetadata,
        do_collapse: bool=False):
    parsimony_estimator = CLTParsimonyEstimator(
            bcode_meta,
            args.out_folder,
            args.mix_path)
    #TODO: DOESN'T USE CELL STATE
    parsimony_trees = parsimony_estimator.estimate(
            obs_leaves,
            num_mix_runs=args.num_jumbles,
            do_collapse=do_collapse)
    logging.info("Total parsimony trees %d", len(parsimony_trees))

    parsimony_score = parsimony_trees[0].get_parsimony_score()
    logging.info("parsimony scores %d", parsimony_score)
    return parsimony_trees

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(seed=args.seed)

    with open(args.obs_data_pkl, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))

    true_model_dict = None
    oracle_measurer = None
    distance_cls = UnrootRFDistanceMeasurer
    if args.true_model_pkl is not None:
        with open(args.true_model_pkl, "rb") as f:
            true_model_dict = six.moves.cPickle.load(f)
            oracle_measurer = distance_cls(true_model_dict["collapsed_subtree"], args.scratch_dir)

    trees_to_output = []

    # Get the parsimony-estimated topologies
    parsimony_trees = get_parsimony_trees(
        obs_leaves,
        args,
        bcode_meta,
        do_collapse=False)

    if args.max_best_multifurc or args.max_best:
        # Get the trees based on distance to the true tree
        oracle_tuples = get_sorted_parsimony_trees(parsimony_trees, oracle_measurer)
        trees_to_output = get_bifurc_multifurc_trees(
                oracle_tuples,
                args.max_best,
                args.max_best_multifurc,
                'best_parsimony',
                distance_cls,
                args.scratch_dir)

    # Just get random trees
    random.shuffle(parsimony_trees)
    random_tree_tuples = [(t, None) for t in parsimony_trees]
    random_trees_to_output = get_bifurc_multifurc_trees(
            random_tree_tuples,
            args.max_random,
            args.max_random_multifurc,
            'random_parsimony',
            distance_cls,
            args.scratch_dir)
    trees_to_output += random_trees_to_output

    # Save each tree as separate pickle file
    for i, tree_topology_dict in enumerate(trees_to_output):
        tree_pickle_out = "%s/parsimony_tree%d.pkl" % (args.out_folder, i)
        print(tree_pickle_out)
        logging.info(tree_topology_dict["tree"].get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        with open(tree_pickle_out, "wb") as f:
            six.moves.cPickle.dump(tree_topology_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
