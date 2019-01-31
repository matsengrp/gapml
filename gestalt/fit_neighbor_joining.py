"""
Neighbor joining tree inference
"""
import sys
import six
import os
import time
import argparse
import logging
import random
import numpy as np

from clt_neighbor_joining_estimator import CLTNeighborJoiningEstimator
from clt_chronos_estimator import CLTChronosEstimator
from common import create_directory
from tune_topology import read_data, read_true_model_files

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
        '--lambdas',
        type=str,
        default='0.01,0.1,1,10,100',
        help='lambdas to use when fitting chronos')
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

    args.lambdas = [float(lam) for lam in args.lambdas.split(",")]
    # we don't have parsimony trees to choose from for NJ, so no topology file
    args.topology_file = None

    create_directory(args.out_model_file)
    create_directory(args.scratch_dir)

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    # Load data
    bcode_meta, _, obs_data_dict = read_data(args)
    orig_num_leaves = len(obs_data_dict['obs_leaves'])
    true_model_dict, assessor = read_true_model_files(
                                    args,
                                    bcode_meta.num_barcodes)

    st_time = time.time()
    neighbor_joining_est = CLTNeighborJoiningEstimator(bcode_meta,
                                                       args.scratch_dir,
                                                       obs_data_dict["obs_leaves"])
    nj_clt = neighbor_joining_est.estimate()
    nj_clt.label_node_ids()
    assert len(nj_clt) == orig_num_leaves

    chronos_est = CLTChronosEstimator(
        nj_clt,
        bcode_meta,
        args.scratch_dir,
        obs_data_dict["time"])
    # Assess the tree if true tree supplied
    dist_dict = None
    if assessor is not None:
        dist_dict = assessor.assess(nj_clt)
    else:
        dist_dict = None
    logging.info("fitted tree: %s", dist_dict)
    results = [{"lambda": None, "fitted_tree": nj_clt, "performance": dist_dict}]

    for lam in args.lambdas:
        nj_clt = chronos_est.estimate(lam)
        assert(len(nj_clt) == orig_num_leaves)

        # Assess the tree if true tree supplied
        dist_dict = None
        if assessor is not None:
            dist_dict = assessor.assess(nj_clt)
            logging.info("fitted tree: lambda %f, %s", lam, dist_dict)

        res = {
            "lambda": lam,
            "fitted_tree": nj_clt,
            "performance": dist_dict
        }
        results.append(res)

    tot_time = time.time() - st_time
    with open(args.out_model_file, "wb") as f:
        six.moves.cPickle.dump(results, f, protocol=2)
    logging.info("Complete!!!")
    logging.info("Total time: %d", tot_time)


if __name__ == "__main__":
    main()
