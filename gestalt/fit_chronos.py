"""
Applies sanderson 2002
"""
import sys
import six
import os
import argparse
import logging
import random
import numpy as np


from clt_chronos_estimator import CLTChronosEstimator
from common import create_directory
from tune_topology import read_data, read_true_model_files, _do_random_rearrange


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

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    # Load data and topology
    bcode_meta, tree, obs_data_dict = read_data(args)
    tree = _do_random_rearrange(tree, bcode_meta, args.num_init_random_rearrange)
    orig_num_leaves = len(tree)
    logging.info("num leaves %d", orig_num_leaves)
    logging.info(tree.get_ascii(attributes=["allele_events_list_str"]))

    true_model_dict, assessor = read_true_model_files(args, bcode_meta.num_barcodes)

    chronos_est = CLTChronosEstimator(
        tree,
        bcode_meta,
        args.scratch_dir,
        obs_data_dict["time"])
    results = []
    for lam in args.lambdas:
        root_clt = chronos_est.estimate(lam)
        assert(len(root_clt) == orig_num_leaves)

        # Assess the tree if true tree supplied
        dist_dict = None
        if assessor is not None:
            dist_dict = assessor.assess(root_clt)
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
