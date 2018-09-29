"""
Neighbor joining tree inference
"""
import sys
import six
import os
import argparse
import logging
import random
import subprocess
import time
import numpy as np

from ete3 import Tree

from cell_lineage_tree import CellLineageTree
from common import create_directory, get_randint, save_data, get_init_target_lams
from clt_likelihood_penalization import mark_target_status_to_penalize
from tune_topology import read_data, read_true_model_files

from Bio.Phylo import draw_ascii, write
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='compute neighbor joining tree')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--out-model-file',
        type=str,
        default="_output/neighbor_joining_fitted.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log_neighbor_joining.txt")
    parser.add_argument(
        '--true-model-file',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default=None)

    parser.set_defaults()
    args = parser.parse_args(args)

    # we don't have parsimony trees to choose from for NJ, so no topology file
    args.topology_file = None

    create_directory(args.out_model_file)

    return args

def _do_convert(tree_node, clt_node):
    for child in tree_node.get_children():
        clt_child = CellLineageTree(
            clt_node.allele_list,
            clt_node.allele_events_list,
            clt_node.cell_state,
            dist=child.dist)
        clt_child.name = child.name
        clt_node.add_child(clt_child)
        _do_convert(child, clt_child)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    # Load data
    bcode_meta, _, obs_data_dict = read_data(args)
    true_model_dict, assessor = read_true_model_files(args, bcode_meta.num_barcodes)

    # construct a right-triangular distance matrix using the cardinality of the
    # symmetric difference of the sets of events in each barcode in two leaves
    distance_matrix = []
    for row, row_leaf in enumerate(obs_data_dict['obs_leaves']):
        distance_matrix_row = []
        for col_leaf in obs_data_dict['obs_leaves'][:(row + 1)]:
            distance_matrix_row.append(sum(len(set(row_allele_events.events) ^ set(col_allele_events.events))
                                           for row_allele_events, col_allele_events in zip(row_leaf.allele_events_list,
                                                                                           col_leaf.allele_events_list)))
        distance_matrix.append(distance_matrix_row)
    distance_matrix = _DistanceMatrix(names=[str(i) for i in range(len(obs_data_dict['obs_leaves']))],
                                      matrix=distance_matrix)
    constructor = DistanceTreeConstructor()
    fitted_tree = constructor.nj(distance_matrix)
    newick_tree_file = '{}/tmp.nk'.format(args.scratch_dir)
    write(fitted_tree, newick_tree_file, 'newick')

    # Read fitted tree into ETE tree
    fitted_tree = Tree(newick_tree_file, format=1)
    # Convert the fitted tree back to a cell lineage tree
    # NOTE: arbitrarily using the first allele in observed leaves to initialize
    #       barcode states. We will later update the leaf states only
    root_clt = CellLineageTree(
            obs_data_dict['obs_leaves'][0].allele_list,
            obs_data_dict['obs_leaves'][0].allele_events_list,
            obs_data_dict['obs_leaves'][0].cell_state,
            dist=0)
    _do_convert(fitted_tree, root_clt)
    # update the leaves to have the correct barcode states
    for leaf in root_clt:
        leaf_parent = leaf.up
        leaf.detach()
        leaf_parent.add_child(CellLineageTree(
                obs_data_dict['obs_leaves'][int(leaf.name)].allele_list,
                obs_data_dict['obs_leaves'][int(leaf.name)].allele_events_list,
                obs_data_dict['obs_leaves'][int(leaf.name)].cell_state,
                dist=leaf.dist))
    logging.info("Done with fitting tree using neighbor joining")
    logging.info(fitted_tree.get_ascii(attributes=["dist"]))

    # Assess the tree if true tree supplied
    dist_dict = None
    if assessor is not None:
        dist_dict = assessor.assess(None, root_clt)
        logging.info("fitted tree: %s", dist_dict)
    results = {
        "fitted_tree": root_clt,
        "performance": dist_dict
    }

    # Save results
    with open(args.out_model_file, "wb") as f:
        six.moves.cPickle.dump(results, f, protocol=2)
    logging.info("Complete!!!")


if __name__ == "__main__":
    main()
