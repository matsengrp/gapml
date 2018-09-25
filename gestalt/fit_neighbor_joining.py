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
    for row, row_leaf in enumerate(obs_data_dict['obs_leaves'][1:], 1):
        distance_matrix_row = []
        for col, col_leaf in enumerate(obs_data_dict['obs_leaves'][:row]):
            distance_matrix_row.append(sum(len(set(row_allele_events.events) ^ set(col_allele_events.events))
                                           for row_allele_events, col_allele_events in zip(row_leaf.allele_events_list,
                                                                                           col_leaf.allele_events_list)))
        distance_matrix.append(distance_matrix_row)
    distance_matrix = _DistanceMatrix(names=[str(i) for i in range(len(obs_data_dict['obs_leaves']))])
    print(distance_matrix)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix)
    print(tree)
    print('so far all I do is print the distance matrix!')
    return

    # Track the leaves by naming them.
    # So we can reconstruct the fitted tree from the R output
    orig_leaf_dict = {}
    for leaf_num, leaf in enumerate(tree):
        leaf.name = str(leaf_num)
        orig_leaf_dict[leaf.name] = leaf

    # Write tree to newick
    suffix = "%d%d" % (int(time.time()), np.random.randint(1000000))
    tree_in_file = "%s/tree_newick%s.txt" % (
            args.scratch_dir, suffix)
    with open(tree_in_file, "w") as f:
        f.write(tree.write(format=5))
        f.write("\n")

    # Call Rscript -- fit the tree to get branch lengths!
    command = 'Rscript'
    script_file = '../R/fit_chronos.R'
    tree_out_file = "%s/fitted_tree%s.txt" % (
            args.scratch_dir, suffix)

    results = []
    for lam in args.lambdas:
        cmd = [
                command,
                script_file,
                tree_in_file,
                tree_out_file,
                str(obs_data_dict["time"]),
                str(lam),
            ]
        print("Calling:", " ".join(cmd))
        res = subprocess.call(cmd)

        # Read fitted tree
        with open(tree_out_file, "r") as f:
            newick_tree = f.readlines()[0]
            fitted_tree = Tree(newick_tree)
        logging.info("Done with fitting tree using chronos, lam %f", lam)
        logging.info(fitted_tree.get_ascii(attributes=["dist"]))

        # Convert the fitted tree back to a cell lineage tree
        root_clt = CellLineageTree(
                tree.allele_list,
                tree.allele_events_list,
                tree.cell_state,
                dist=0)
        _do_convert(fitted_tree, root_clt)
        for leaf in root_clt:
            leaf_parent = leaf.up
            leaf.detach()
            orig_cell_lineage_tree = orig_leaf_dict[leaf.name]
            orig_cell_lineage_tree.dist = leaf.dist
            orig_cell_lineage_tree.detach()
            leaf_parent.add_child(orig_cell_lineage_tree)

        # Assess the tree if true tree supplied
        dist_dict = None
        if assessor is not None:
            dist_dict = assessor.assess(None, root_clt)
            logging.info("fitted tree: %s", dist_dict)
        res = {
            "lambda": lam,
            "fitted_tree": root_clt,
            "performance": dist_dict
        }
        results.append(res)

    # Save results
    with open(args.out_model_file, "wb") as f:
        six.moves.cPickle.dump(results, f, protocol=2)
    logging.info("Complete!!!")


if __name__ == "__main__":
    main()
