"""
A simulation engine to test our hillclimbing
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
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import pickle
from pathlib import Path
from ete3 import Tree

import ancestral_events_finder as anc_evt_finder
from cell_state import CellState, CellTypeTree
from cell_state_simulator import CellTypeSimulator
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator
from clt_likelihood_estimator import *
from alignment import AlignerNW
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_manipulation import resolve_all_multifurcs
from clt_likelihood_topology import CLTLikelihoodTopologySearcher

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
        '--num-jobs',
        type=int,
        default=1,
        help="number of jobs")
    parser.add_argument(
        '--max-nni-steps',
        type=int,
        default=1,
        help="number of nni steps to take to explore around for hill climbing")
    parser.add_argument(
        '--num-nni-restarts',
        type=int,
        default=5,
        help="number of times we restart the nni local search for candidate trees")
    parser.add_argument(
        '--max-tree-search-iters',
        type=int,
        default=30,
        help="number of iterations for tree search")
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
            default=1)
    parser.add_argument('--topology-only', action='store_true', help="topology only")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    args.fitted_models_file = "%s/fitted.pkl" % args.out_folder
    args.scratch_dir = os.path.join(args.out_folder, "likelihood%s" % int(time.time()))
    print("Log file", args.log_file)
    print("Scratch dir", args.scratch_dir)

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

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

        # Oracle tree
        oracle_pen_ll, oracle_model = fit_pen_likelihood(
                true_tree,
                args,
                bcode_meta,
                cell_type_tree,
                approximator,
                sess)
        # Use this as a reference
        logging.info("oracle %f", oracle_pen_ll)

        # Begin our hillclimbing search!
        # Get the parsimony-estimated topologies
        parsimony_estimator = CLTParsimonyEstimator(
                bcode_meta,
                args.out_folder,
                args.mix_path)
        #TODO: DOESN'T USE CELL STATE
        parsimony_trees = parsimony_estimator.estimate(
                obs_leaves,
                num_mix_runs = args.num_jumbles,
                do_collapse = True)
        # Just take the first parsimony tree for now
        parsimony_tree = parsimony_trees[0]
        # Make the parsimony tree a bifurcating tree to initialize our topology search
        resolve_all_multifurcs(parsimony_tree)
        logging.info("Parsimony tree")
        logging.info(parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        logging.info(parsimony_tree.get_ascii(attributes=["observed"], show_internal=True))
        logging.info(parsimony_tree.get_ascii(attributes=["anc_state_list_str"], show_internal=True))

        topo_searcher = CLTLikelihoodTopologySearcher(
                bcode_meta,
                cell_type_tree if args.use_cell_state else None,
                args.know_cell_lambdas,
                np.array(args.target_lambdas) if args.know_target_lambdas else None,
                args.log_barr,
                args.max_iters,
                approximator,
                sess,
                args.scratch_dir,
                true_tree = true_tree,
                do_distributed = args.num_jobs > 1)
        topo_searcher.search(
                parsimony_tree,
                max_iters=args.max_tree_search_iters,
                num_nni_restarts=args.num_nni_restarts,
                max_nni_steps=args.max_nni_steps)

if __name__ == "__main__":
    main()
