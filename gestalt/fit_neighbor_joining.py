"""
Neighbor joining tree inference
"""
import sys
import six
import os
import argparse
import logging
import random
import numpy as np

from clt_neighbor_joining_estimator import CLTNeighborJoiningEstimator
from common import create_directory
from tune_topology import read_data, read_true_model_files, _do_random_rearrange
from tree_distance import UnrootRFDistanceMeasurer

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


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    # Load data
    bcode_meta, _, obs_data_dict = read_data(args)
    neighbor_joining_est = CLTNeighborJoiningEstimator(bcode_meta,
                                                       args.scratch_dir,
                                                       obs_data_dict["obs_leaves"])
    root_clt = neighbor_joining_est.estimate()

    true_model_dict, assessor = read_true_model_files(
                                    args,
                                    bcode_meta.num_barcodes,
                                    measurer_classes=[UnrootRFDistanceMeasurer])

    assert len(root_clt) == len(obs_data_dict['obs_leaves'])

    # Assess the tree if true tree supplied
    dist_dict = None
    if assessor is not None:
        dist_dict = assessor.assess(root_clt)
        logging.info("fitted tree: %s", dist_dict)
        results = {"fitted_tree": root_clt, "performance": dist_dict}
        with open(args.out_model_file, "wb") as f:
            six.moves.cPickle.dump(results, f, protocol=2)
    logging.info("Complete!!!")



if __name__ == "__main__":
    main()
