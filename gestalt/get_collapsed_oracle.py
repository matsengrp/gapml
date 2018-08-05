"""
Get tree topologies based on the oracle
"""
from __future__ import division, print_function
import os
import sys
import argparse
import logging
from pathlib import Path
import six

from tree_distance import *
from collapsed_tree import collapse_zero_lens
from common import create_directory

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
        '--out-template-file',
        type=str,
        default="_output/oracle_tree0.pkl",
        help='template file name for outputs. this code will replace 0 with other tree indices')

    args = parser.parse_args()
    args.out_folder = os.path.dirname(args.out_template_file)
    assert os.path.join(args.out_folder, "oracle_tree0.pkl") == args.out_template_file
    create_directory(args.out_template_file)

    args.log_file = "%s/oracle_tree_log.txt" % args.out_folder
    print("Log file", args.log_file)
    return args

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

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.true_model_pkl, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)

    #oracle_multifurc_tree = collapse_internally_labelled_tree(true_model_dict["collapsed_subtree"])
    #logging.info(true_model_dict["collapsed_subtree"].get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    #logging.info(oracle_multifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    trees_to_output = [{
        'selection_type': 'oracle',
        'multifurc': False,
        'idx': 0,
        'aux': None,
        'tree': true_model_dict["true_subtree"],
    },
    #{
    #    'selection_type': 'oracle',
    #    'multifurc': True,
    #    'idx': 0,
    #    'aux': None,
    #    'tree': oracle_multifurc_tree}
    ]

    # Save each tree as separate pickle file
    for i, tree_topology_dict in enumerate(trees_to_output):
        tree_pickle_out = "%s/oracle_tree%d.pkl" % (args.out_folder, i)
        print(tree_pickle_out)
        with open(tree_pickle_out, "wb") as f:
            six.moves.cPickle.dump(tree_topology_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
