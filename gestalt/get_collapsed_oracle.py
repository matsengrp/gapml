"""
Get tree topologies based on the oracle
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
import argparse
import time
import random
import tensorflow as tf
import logging
from pathlib import Path
import six

from tree_distance import *
from constants import *
from common import *
from simulate_common import *

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
        default="_output/true_model.pkl",
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

    args = parser.parse_args()
    args.log_file = "%s/oracle_tree_log.txt" % args.out_folder
    print("Log file", args.log_file)
    return args


def get_sorted_parsimony_trees(parsimony_trees, oracle_measurer):
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
        tree_list,
        max_bifurc,
        max_multifurc,
        selection_type,
        distance_cls,
        scratch_dir):
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

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(seed=args.seed)

    with open(args.true_model_pkl, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)

    oracle_coll_tree = collapse_internally_labelled_tree(true_model_dict["true_tree"])
    logging.info(true_model_dict["true_tree"].get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(oracle_coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    trees_to_output = [{
        'selection_type': 'oracle',
        'multifurc': False,
        'idx': 0,
        'aux': None,
        'tree': true_model_dict["true_tree"],
    }, {
        'selection_type': 'oracle',
        'multifurc': True,
        'idx': 0,
        'aux': None,
        'tree': oracle_coll_tree}]

    # Save each tree as separate pickle file
    for i, tree_topology_dict in enumerate(trees_to_output):
        tree_pickle_out = "%s/oracle_tree%d.pkl" % (args.out_folder, i)
        print(tree_pickle_out)
        with open(tree_pickle_out, "wb") as f:
            six.moves.cPickle.dump(tree_topology_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
