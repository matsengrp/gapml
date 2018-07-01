"""
Apply filtration -- we can only observe the first `n` alleles in each cell.
"""
from __future__ import division, print_function
import os
import sys
import argparse
import logging
import six
from typing import Dict, List

import collapsed_tree
from cell_lineage_tree import CellLineageTree
from clt_observer import ObservedAlignedSeq
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

def collapse_tree_by_first_alleles(true_subtree: CellLineageTree, num_barcodes: int):
    """
    Generate the collapsed tree based on first `num_barcodes` alleles
    @return CellLineageTree
    """
    # Truncate the number of alleles observed at the internal nodes
    for node in true_subtree.traverse():
        node.set_allele_list(node.allele_list.create_truncated_version(num_barcodes))
    # Create the collapsed CLT according to the first `args.num_barcodes` alleles
    return collapsed_tree.collapse_ultrametric(true_subtree)

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.info(str(args))

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    bcode_meta = obs_data_dict["bcode_meta"]
    assert args.num_barcodes <= bcode_meta.num_barcodes

    # Update the barcode metadata to have the collapsed number of barcodes
    bcode_meta.num_barcodes = args.num_barcodes

    # Now start collapsing the observations by first n alleles
    obs_leaves = obs_data_dict["obs_leaves"]
    obs_data_dict["obs_leaves"] = collapse_obs_leaves_by_first_alleles(
            obs_leaves,
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
            true_model_dict["true_subtree"],
            args.num_barcodes)
    save_data(collapsed_clt, args.out_collapsed_tree_file)

    # Plot the MRCA matrix of the true collapsed tree for fun
    out_png = args.out_collapsed_tree_file.replace(".pkl", "_mrca.png")
    plot_mrca_matrix(collapsed_clt, out_png)

if __name__ == "__main__":
    main()
