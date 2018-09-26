"""
Applies sanderson 2002
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
import ancestral_events_finder
from clt_likelihood_penalization import mark_target_status_to_penalize
from tune_topology import read_data, read_true_model_files, _do_random_rearrange
from collapsed_tree import _remove_single_child_unobs_nodes


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
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
        '--topology-file',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help="Topology file")
    parser.add_argument(
        '--out-model-file',
        type=str,
        default="_output/chronos_fitted.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log_chronos.txt")
    parser.add_argument(
        '--true-model-file',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--lambdas',
        type=str,
        default='0.01,0.1,1,10,100',
        help='lambdas to use when fitting chronos')
    parser.add_argument(
        '--num-init-random-rearrange',
        type=int,
        default=0,
        help='number of times we randomly rearrange tree at the beginning')
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default=None)

    parser.set_defaults()
    args = parser.parse_args(args)

    create_directory(args.out_model_file)
    if args.scratch_dir is None:
        topology_folder = os.path.dirname(args.topology_file)
        args.scratch_dir = os.path.join(topology_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.lambdas = [float(lam) for lam in args.lambdas.split(",")]
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

    # Load data and topology
    bcode_meta, tree, obs_data_dict = read_data(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    tree = _do_random_rearrange(tree, bcode_meta, args.num_init_random_rearrange)
    _remove_single_child_unobs_nodes(tree)
    if len(tree.get_children()) == 1:
        tree.get_children()[0].delete()

    true_model_dict, assessor = read_true_model_files(args, bcode_meta.num_barcodes)
    ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
    parsimony_score = ancestral_events_finder.get_parsimony_score(tree)

    logging.info(tree.get_ascii(attributes=["allele_events_list_str"]))
    logging.info(tree.get_ascii(attributes=["dist"]))

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
            logging.info("fitted tree: lambda %f, %s", lam, dist_dict)
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
