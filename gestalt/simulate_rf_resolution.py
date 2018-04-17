"""
A simulation engine to see how well cell lineage estimation performs
"""
from __future__ import division, print_function
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import pickle
from pathlib import Path

from cell_state import CellState, CellTypeTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulatorBifurcating
from clt_simulator_simple import CLTSimulatorOneLayer, CLTSimulatorTwoLayers
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator
from clt_likelihood_estimator import *
from alignment import AlignerNW
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_manipulation import search_nearby_trees

from constants import *
from common import *
from summary_util import *
from simulate_common import *

def parse_args():
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        '--out-folder',
        type=str,
        default="_output",
        help='folder to put output in')
    parser.add_argument(
        '--num-moves',
        type=int,
        default=20,
        help="number of trees to consider per rf dist")
    parser.add_argument(
        '--num-explore-trees',
        type=int,
        default=5,
        help="number of trees to consider per rf dist")
    parser.add_argument(
        '--num-barcodes',
        type=int,
        default=1,
        help="number of independent barcodes. we assume all the same")
    parser.add_argument(
        '--target-lambdas',
        type=float,
        nargs=2,
        default=[0.01] * 10,
        help='target cut rates -- will get slightly perturbed for the true value')
    parser.add_argument(
        '--know-target-lambdas',
        action='store_true')
    parser.add_argument(
        '--know-cell-lambdas',
        action='store_true')
    parser.add_argument(
        '--const-branch-len',
        action='store_true')
    parser.add_argument(
        '--repair-long-probability',
        type=float,
        nargs=2,
        default=[0.001] * 2,
        help='probability of doing no deletion/insertion during repair')
    parser.add_argument(
        '--repair-indel-probability',
        type=float,
        default=0.1,
        help='probability of doing no deletion/insertion during repair')
    parser.add_argument(
        '--repair-deletion-lambda',
        type=float,
        default=3,
        help=
        'poisson parameter for distribution of symmetric deltion about cut site(s)'
    )
    parser.add_argument(
        '--repair-insertion-lambda',
        type=float,
        default=1,
        help='poisson parameter for distribution of insertion in cut site(s)')
    parser.add_argument(
        '--birth-lambda', type=float, default=2, help='birth rate')
    parser.add_argument(
        '--death-lambda', type=float, default=0.001, help='death rate')
    parser.add_argument(
        '--time', type=float, default=1.2, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.9,
        help='proportion cells sampled/alleles successfully sequenced')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument(
        '--single-layer', action='store_true', help='single layer tree')
    parser.add_argument(
        '--two-layers', action='store_true', help='two layer tree')
    parser.add_argument(
            '--model-seed',
            type=int,
            default=0,
            help="Seed for generating the model")
    parser.add_argument(
            '--data-seed',
            type=int,
            default=0,
            help="Seed for generating data")
    parser.add_argument(
            '--log-barr',
            type=float,
            default=0.2,
            help="log barrier parameter on the branch lengths")
    parser.add_argument(
            '--lasso-param',
            type=float,
            default=0,
            help="lasso parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=2000)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=100)
    parser.add_argument('--max-clt-nodes', type=int, default=8000)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--use-cell-state', action='store_true')
    parser.add_argument('--max-trees',
            type=int,
            default=2)
    parser.add_argument('--num-jumbles',
            type=int,
            default=10)
    parser.add_argument('--topology-only', action='store_true', help="topology only")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    args.fitted_models_file = "%s/fitted.pkl" % args.out_folder
    args.branch_plot_file = "%s/branch_lens.png" % args.out_folder

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(seed=args.model_seed)

    barcode_orig = BarcodeMetadata.create_fake_barcode_str(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
    bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig, num_barcodes = args.num_barcodes)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    args.target_lambdas = np.array(args.target_lambdas) + np.random.uniform(size=args.num_targets) * 0.08
    logging.info("args.target_lambdas %s" % str(args.target_lambdas))

    # Create a cell-type tree
    cell_type_tree = create_cell_type_tree(args)

    logging.info(str(args))

    sess = tf.Session()
    with sess.as_default():
        # Create model
        clt_model = CLTLikelihoodModel(
                None,
                bcode_meta,
                sess,
                target_lams = np.array(args.target_lambdas),
                trim_long_probs = np.array(args.repair_long_probability),
                trim_zero_prob = args.repair_indel_probability,
                trim_poissons = np.array([args.repair_deletion_lambda, args.repair_deletion_lambda]),
                insert_zero_prob = args.repair_indel_probability,
                insert_poisson = args.repair_insertion_lambda,
                cell_type_tree = cell_type_tree)
        clt_model.set_tot_time(args.time)
        tf.global_variables_initializer().run()

        clt, obs_leaves, true_tree = create_cell_lineage_tree(args, clt_model)
        # Gather true branch lengths
        true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)
        true_branch_lens = {}
        for node in true_tree.traverse(CLTLikelihoodModel.NODE_ORDER):
            if not node.is_root():
                true_branch_lens[node.node_id] = node.dist

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)
        nearby_trees = []
        for _ in range(3):
            nearby_trees += search_nearby_trees(true_tree, max_search_dist=args.num_moves)
        nearby_tree_dict = get_rf_dist_dict(
                nearby_trees,
                true_tree)

        # Fit trees
        fitting_results = []
        for rf_dist, trees in nearby_tree_dict.items():
            logging.info(
                    "There are %d trees with RF %d",
                    len(trees),
                    rf_dist)
            for tree, rooted_rf, unrooted_rf in trees[:args.num_explore_trees]:
                pen_log_lik, res_model = fit_pen_likelihood(
                        tree,
                        args,
                        bcode_meta,
                        cell_type_tree,
                        approximator,
                        sess)
                fitting_results.append((
                    pen_log_lik,
                    rooted_rf,
                    unrooted_rf,
                    res_model))

                # Print some summaries
                logging.info("pen log lik %f RF %d", pen_log_lik, rf_dist)

        # Correlation between RF dist and likelihood among parsimony trees
        pen_log_lik, oracle_model = fit_pen_likelihood(
                true_tree,
                args,
                bcode_meta,
                cell_type_tree,
                approximator,
                sess)
        logging.info("True tree score %f", pen_log_lik)
        fitting_results.append((pen_log_lik, 0, 0, oracle_model))

        unrooted_rf_dists = []
        rooted_rf_dists = []
        pen_log_liks = []
        for pen_ll, rooted_rf, unrooted_rf, _ in fitting_results:
            unrooted_rf_dists.append(unrooted_rf)
            rooted_rf_dists.append(rooted_rf)
            pen_log_liks.append(pen_ll[0])
        logging.info("unroot rf_dists %s", str(unrooted_rf_dists))
        logging.info("root rf_dists %s", str(rooted_rf_dists))
        logging.info("pen log liks %s", str(pen_log_liks))
        logging.info("unroot pearson rf to log lik %s", pearsonr(unrooted_rf_dists, pen_log_liks))
        logging.info("unroot spearman rf to log lik %s", spearmanr(unrooted_rf_dists, pen_log_liks))
        plt.scatter(unrooted_rf_dists, pen_log_liks)
        plt.savefig("%s/unroot_rf_dist_to_ll.png" % args.out_folder)
        logging.info("root pearson rf to log lik %s", pearsonr(rooted_rf_dists, pen_log_liks))
        logging.info("root spearman rf to log lik %s", spearmanr(rooted_rf_dists, pen_log_liks))
        plt.scatter(rooted_rf_dists, pen_log_liks)
        plt.savefig("%s/root_rf_dist_to_ll.png" % args.out_folder)

        # Fit oracle tree
        save_fitted_models(args.fitted_models_file, fitting_results)

if __name__ == "__main__":
    main()
