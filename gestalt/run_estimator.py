"""
This code fits model parameters and branch lengths
constrained to the multifurcating tree topology.

Searches over bifurcating trees with the same multifurcating tree
by using a continuous parameterization.
Suppose constant events when resolving multifurcations.
"""
from __future__ import division, print_function
import os
import sys
import csv
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr
import logging
import six

from transition_wrapper_maker import TransitionWrapperMaker
from likelihood_scorer import LikelihoodScorer
from plot_mrca_matrices import plot_mrca_matrix
from tree_distance import *
from constants import *
from common import *

def parse_args():
    parser = argparse.ArgumentParser(description='fit topology and branch lengths for GESTALT')
    parser.add_argument(
        '--obs-data',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--tree-topology-pkl',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help='pkl file with tree topology')
    parser.add_argument(
        '--true-model-pkl',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--out-folder',
        type=str,
        default="_output",
        help='folder to put output in')
    parser.add_argument(
        '--time', type=float, default=1.2, help='how much time to fit for')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument(
            '--seed',
            type=int,
            default=40)
    parser.add_argument(
            '--log-barr',
            type=float,
            default=0.001,
            help="log barrier parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--num-inits', type=int, default=1)

    parser.set_defaults(use_cell_state=False)
    args = parser.parse_args()
    args.log_file = args.tree_topology_pkl.replace(".pkl", "_fit_log.txt")
    print("Log file", args.log_file)
    args.pickle_out = args.tree_topology_pkl.replace(".pkl", "_fitted.pkl")
    args.csv_out = args.tree_topology_pkl.replace(".pkl", "_fitted.csv")
    args.scratch_dir = os.path.join(args.out_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(seed=args.seed)

    with open(args.obs_data, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))

    with open(args.tree_topology_pkl, "rb") as f:
        tree_topology_dict = six.moves.cPickle.load(f)
        tree = tree_topology_dict["tree"]
    logging.info("Tree topology info: %s", tree_topology_dict)

    true_model_dict = None
    oracle_dist_measurers = None
    if args.true_model_pkl is not None:
        with open(args.true_model_pkl, "rb") as f:
            true_model_dict = six.moves.cPickle.load(f)
            oracle_dist_measurers = TreeDistanceMeasurerAgg([
                UnrootRFDistanceMeasurer,
                RootRFDistanceMeasurer,
                #SPRDistanceMeasurer,
                MRCADistanceMeasurer],
                true_model_dict["collapsed_subtree"],
                args.scratch_dir)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    transition_wrap_maker = TransitionWrapperMaker(tree, bcode_meta)

    worker = LikelihoodScorer(
       args.seed,
       tree,
       bcode_meta,
       None, # Do not use cell type info
       False, # Do not know cell type lambdas
       None, # Do not know target lambdas
       args.log_barr,
       args.max_iters,
       transition_wrap_maker,
       tot_time = args.time,
       dist_measurers = oracle_dist_measurers)

    res = worker.do_work_directly(sess)
    print(res.model_params_dict)
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["node_id"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["cell_state"]))

    if oracle_dist_measurers is not None:
        result_print_dict = oracle_dist_measurers.get_tree_dists([res.fitted_bifurc_tree])[0]
        true_target_lambdas = true_model_dict["true_model_params"]["target_lams"]
        pearson_target = pearsonr(
                true_target_lambdas,
                res.model_params_dict["target_lams"])
        result_print_dict["pearson_target_corr"] = pearson_target[0]
        result_print_dict["pearson_target_pval"] = pearson_target[1]
        result_print_dict["target_lam_dist"] = np.linalg.norm(
                true_target_lambdas - res.model_params_dict["target_lams"])
    else:
        result_print_dict = {}
    result_print_dict["selection_type"] = tree_topology_dict["selection_type"]
    result_print_dict["log_lik"] = res.train_history[-1]["log_lik"][0]

    # Save distance data as csv
    with open(args.csv_out, 'w') as csvfile:
        fieldnames = list(result_print_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerow(result_print_dict)

    # Save the data
    with open(args.pickle_out, "wb") as f:
        six.moves.cPickle.dump(res, f, protocol = 2)

    plot_mrca_matrix(
        res.fitted_bifurc_tree,
        args.pickle_out.replace(".pkl", "_mrca.png"))

if __name__ == "__main__":
    main()
