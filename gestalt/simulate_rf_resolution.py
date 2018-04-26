"""
A simulation engine to see how log likelihood correlates with rf distances
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr
import logging
import pickle

from clt_estimator import CLTEstimator
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_manipulation import search_nearby_trees
import ancestral_events_finder as anc_evt_finder
from likelihood_scorer import LikelihoodScorer

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
        default=2,
        help="number of steps to perturb away from the true tree")
    parser.add_argument(
        '--num-searches',
        type=int,
        default=1,
        help="number of times to restart the tree perturbations")
    parser.add_argument(
        '--num-explore-trees',
        type=int,
        default=2,
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
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=100)
    parser.add_argument('--max-clt-nodes', type=int, default=8000)
    parser.add_argument('--use-cell-state', action='store_true')
    parser.add_argument('--do-distributed', action='store_true', help="submit slurm jobs")

    parser.set_defaults(single_layer=False, two_layers=False)
    args = parser.parse_args()

    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/rf_resolution_log.txt" % args.out_folder
    args.fitted_models_file = "%s/rf_resolution_results.pkl" % args.out_folder
    args.scratch_dir = os.path.join(args.out_folder, "likelihood%s" % int(time.time()))
    print("Log file", args.log_file)
    print("scratch dir", args.scratch_dir)

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
        clt_model.tot_time = args.time
        tf.global_variables_initializer().run()

        # Simulate data
        clt, obs_leaves, true_tree = create_cell_lineage_tree(args, clt_model)
        # Process tree by labeling nodes in the tree
        anc_evt_finder.annotate_ancestral_states(true_tree, bcode_meta)
        true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)

        # Perform NNI moves to find nearby trees
        nearby_trees = []
        for i in range(args.num_searches):
            logging.info("Searching nearby trees, search %d", i)
            nearby_trees += search_nearby_trees(true_tree, max_search_dist=args.num_moves)
        assert len(nearby_trees) > 0
        nearby_tree_dict = get_rf_dist_dict(nearby_trees, true_tree, unroot=False)

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

        # Fit trees
        worker_list = []
        rf_dists = []
        seed = 0
        for rf_dist, tree_tuples in nearby_tree_dict.items():
            uniq_trees = CLTEstimator.get_uniq_trees(
                    [t[0] for t in tree_tuples],
                    attr_str="allele_events_list_str",
                    max_trees=args.num_explore_trees,
                    unrooted=False)
            logging.info("There are %d trees with RF %d", len(uniq_trees), rf_dist)
            for tree in uniq_trees:
                seed += 1
                lik_scorer = LikelihoodScorer(
                        seed,
                        tree,
                        bcode_meta,
                        cell_type_tree,
                        args.know_cell_lambdas,
                        np.array(args.target_lambdas) if args.know_target_lambdas else None,
                        args.log_barr,
                        args.max_iters,
                        approximator,
                        tot_time = args.time)
                worker_list.append(lik_scorer)
                rf_dists.append(rf_dist)
        assert len(rf_dists) > 2

        if args.do_distributed and len(worker_list) > 1:
            # Submit jobs to slurm
            batch_manager = BatchSubmissionManager(
                    worker_list=worker_list,
                    shared_obj=None,
                    # Each tree is its separate slurm job
                    num_approx_batches=len(worker_list),
                    worker_folder=args.scratch_dir)
            fitting_results = batch_manager.run()
        else:
            # Run jobs locally
            fitting_results = [worker.do_work_directly(sess) for worker in worker_list]

        # Print some summaries
        pen_log_liks = []
        for res, rf_dist in zip(fitting_results, rf_dists):
            pen_ll = res[0][0]
            pen_log_liks.append(pen_ll)
            logging.info("pen log lik %f RF %d", pen_ll, rf_dist)

        # Add in the oracle tree
        oracle_scorer = LikelihoodScorer(
                0, # seed
                true_tree,
                bcode_meta,
                cell_type_tree,
                args.know_cell_lambdas,
                np.array(args.target_lambdas) if args.know_target_lambdas else None,
                args.log_barr,
                args.max_iters,
                approximator,
                tot_time = args.time)
        oracle_res = oracle_scorer.do_work_directly(sess)
        logging.info("True tree score %f", oracle_res[0])
        fitting_results.append(oracle_res)
        pen_log_liks.append(oracle_res[0][0])
        rf_dists.append(0)

        # Correlation between RF dist and likelihood among parsimony trees
        logging.info("rooted rf_dists %s", str(rf_dists))
        logging.info("pen log liks %s", str(pen_log_liks))
        logging.info("pearson rf to log lik %s", pearsonr(rf_dists, pen_log_liks))
        logging.info("spearman rf to log lik %s", spearmanr(rf_dists, pen_log_liks))
        plt.scatter(rf_dists, pen_log_liks)
        plt.savefig("%s/rf_dist_to_ll.png" % args.out_folder)

        with open(args.fitted_models_file, "wb"):
            pickle.dumps({
                "fitting_results": fitting_results,
                "rf_dists": rf_dists,
                "pen_ll": pen_ll})

if __name__ == "__main__":
    main()
