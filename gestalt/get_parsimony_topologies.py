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
from typing import List
import glob

from cell_lineage_tree import CellLineageTree
from clt_observer import ObservedAlignedSeq
from barcode_metadata import BarcodeMetadata
from tree_distance import TreeDistanceMeasurer, UnrootRFDistanceMeasurer
from clt_estimator import CLTParsimonyEstimator
from collapsed_tree import collapse_zero_lens
from constants import *
from constant_paths import *

def parse_args():
    parser = argparse.ArgumentParser(description='generate possible tree topologies using MIX')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--model-file',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/parsimony_log.txt")
    parser.add_argument(
        '--out-template-file',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help='template file name for outputs. this code will replace 0 with other tree indices')
    parser.add_argument(
        '--seed',
        type=int,
        default=5,
        help="Random number generator seed. Also used as seed for MIX. Must be odd")
    parser.add_argument(
        '--num-jumbles',
        type=int,
        default=1,
        help="Number of times to jumble. (This is an input to MIX)")
    parser.add_argument(
        '--mix-path',
        type=str,
        default=MIX_PATH)
    parser.add_argument('--max-random',
        type=int,
        default=1,
        help="""
            Output `max-random` bifurcating trees estimated from parsimony
            Random selection among those estimated from parsimony.
            """)
    parser.add_argument('--max-random-multifurc',
        type=int,
        default=1,
        help="""
            Output `max-random-multifurc` different multifurcating trees estimated from parsimony,
            i.e. collapse the bifurcating trees and output the multifurcating versions.
            """)
    parser.add_argument('--max-best',
        type=int,
        default=0,
        help="""
            Output the top `max-best` bifurcating trees from parsimony.
            Right now, 'best' is measured by distance to the collapsed oracle tree.
            """)
    parser.add_argument('--max-best-multifurc',
        type=int,
        default=0,
        help="""
            Output the top `max-best` multifurcating trees from parsimony.
            Right now, 'best' is measured by distance to the collapsed oracle tree.
            So we find the closest bifurcating trees and then collapse them.
            """)

    args = parser.parse_args()
    assert args.seed % 2 == 1
    if args.max_best_multifurc or args.max_best:
        # Require having true tree to know what is a "best" tree
        assert args.model_file is not None

    args.out_folder = os.path.dirname(args.out_template_file)
    assert os.path.join(args.out_folder, "parsimony_tree0.pkl") == args.out_template_file

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
    @param parsimony_trees: trees from MIX
    @param oracle_measurer: distance measurer to the oracle tree

    @return a sorted list of tree tuples (tree, tree dist) based on
            distance to the true tree
    """
    assert oracle_measurer is not None

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
            coll_tree = collapse_zero_lens(tree)
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
    logging.info("Number of multifurcating trees selected, %s : %d", selection_type, num_multifurc)
    return trees_to_output

def get_parsimony_trees(
        obs_leaves: List[ObservedAlignedSeq],
        args,
        bcode_meta: BarcodeMetadata):
    """
    Run MIX to get maximum parsimony trees
    """
    parsimony_estimator = CLTParsimonyEstimator(
            bcode_meta,
            args.out_folder,
            args.mix_path)
    parsimony_trees = parsimony_estimator.estimate(
            obs_leaves,
            mix_seed=args.seed,
            num_jumbles=args.num_jumbles)
    logging.info("Total parsimony trees %d", len(parsimony_trees))

    parsimony_score = parsimony_trees[0].get_parsimony_score()
    logging.info("parsimony scores %d", parsimony_score)
    return parsimony_trees

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(seed=args.seed)

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))

    true_model_dict = None
    oracle_measurer = None
    distance_cls = UnrootRFDistanceMeasurer
    if args.model_file is not None:
        # TODO: we might not think that the best tree is the collapsed tree. Need to add more flexibility
        #       if we want to use a different tree metric and oracle tree
        with open(args.model_file, "rb") as f:
            true_model_dict = six.moves.cPickle.load(f)
            oracle_measurer = distance_cls(true_model_dict["collapsed_subtree"], args.scratch_dir)

    trees_to_output = []

    # Get the parsimony-estimated topologies
    parsimony_trees = get_parsimony_trees(
        obs_leaves,
        args,
        bcode_meta)

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

    # Remove existing parsimony trees
    topology_files = glob.glob("%s/parsimony_tree*[0-9]*" % args.out_folder)
    for t in topology_files:
        print('remove %s' % t)
        os.remove(t)

    # Save each tree as separate pickle file
    for i, tree_topology_dict in enumerate(trees_to_output):
        tree_pickle_out = "%s/parsimony_tree%d.pkl" % (args.out_folder, i)
        logging.info(tree_pickle_out)
        logging.info(tree_topology_dict)

        logging.info(tree_topology_dict["tree"].get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        with open(tree_pickle_out, "wb") as f:
            six.moves.cPickle.dump(tree_topology_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
