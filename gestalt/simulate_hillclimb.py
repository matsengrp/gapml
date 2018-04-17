"""
A simulation engine to test our hillclimbing
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

import ancestral_events_finder as anc_evt_finder
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
from parallel_worker import BatchSubmissionManager
from likelihood_scorer import LikelihoodScorer

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
    print("Log file", args.log_file)
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    args.fitted_models_file = "%s/fitted.pkl" % args.out_folder
    args.branch_plot_file = "%s/branch_lens.png" % args.out_folder

    return args

def assign_branch_lens(clt_model, tree):
    # Assign branch lengths to this current tree
    curr_br_lens = clt_model.get_branch_lens()
    for node in tree.traverse():
        if not node.is_root():
            node.dist = curr_br_lens[node.node_id]

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

        # Get the parsimony-estimated topologies
        parsimony_estimator = CLTParsimonyEstimator(
                bcode_meta,
                args.out_folder,
                args.mix_path)
        #TODO: DOESN'T USE CELL STATE
        parsimony_trees = parsimony_estimator.estimate(
                obs_leaves,
                num_mix_runs=args.num_jumbles)
        # Just take the first parsimony tree for now?
        parsimony_tree = parsimony_trees[0]
        rf_res = true_tree.robinson_foulds(
                parsimony_tree,
                attr_t1="allele_events_list_str",
                attr_t2="allele_events_list_str",
                expand_polytomies=False,
                unrooted_trees=False)
        assert rf_res[0] > 0

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)
        anc_evt_finder.annotate_ancestral_states(parsimony_tree, bcode_meta)
            
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

        # Now start exploring the sapce with NNI moves
        curr_tree = parsimony_tree
        curr_pen_ll, curr_model = fit_pen_likelihood(
                parsimony_tree,
                args,
                bcode_meta,
                cell_type_tree,
                approximator,
                sess)
        # Assign branch lengths to this current tree
        assign_branch_lens(curr_model, curr_tree)
        curr_model_vars = curr_model.get_vars_as_dict()

        # Begin our hillclimbing search!
        print("Parsimony tree")
        print(parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        logging.info("parsimony init distance %d, init pen_ll %f", rf_res[0], curr_pen_ll)
        for i in range(args.max_tree_search_iters):
            logging.info("curr treeeee....")
            logging.info(curr_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

            # Search around with NNI
            nearby_trees = []
            for _ in range(args.num_nni_restarts):
                nearby_trees += search_nearby_trees(curr_tree, max_search_dist=args.max_nni_steps)
            # uniq ones please
            nearby_trees = CLTEstimator.get_uniq_trees(nearby_trees)

            for tree in nearby_trees:
                logging.info("considering...")
                logging.info(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

            # TODO: distribute
            worker_list = [
                LikelihoodScorer(
                        tree,
                        args,
                        bcode_meta,
                        cell_type_tree,
                        approximator,
                        curr_model_vars)
                for tree in nearby_trees]
            shared_obj = None
            if args.num_jobs > 1:
                batch_manager = BatchSubmissionManager(
                        worker_list,
                        shared_obj,
                        self.num_jobs,
                        os.path.join(self.out_dir, "likelihood"))
                nni_results = batch_manager.run()
            else:
                nni_results = [worker.run(shared_obj) for worker in worker_list]

            # Now let's compare their pen log liks
            pen_lls = [r[0] for r in nni_results]
            best_index = np.argmax(pen_lls)
            if pen_lls[best_index] < curr_pen_ll:
                # None of these are better than the current tree.
                continue

            # We found a tree with higher pen log lik than current tree
            curr_pen_ll = pen_lls[best_index]
            curr_tree = nearby_trees[best_index]
            curr_model = nni_results[best_index][1]

            # Store info about our best current tree for warm starting later on
            curr_model_vars = curr_model.get_vars_as_dict()
            assign_branch_lens(curr_model, curr_tree)

            # Calculate RF distance to understand if our hillclimbing is working
            root_rf_res = true_tree.robinson_foulds(
                    curr_tree,
                    attr_t1="allele_events_list_str",
                    attr_t2="allele_events_list_str",
                    expand_polytomies=False,
                    unrooted_trees=False)
            unroot_rf_res = true_tree.robinson_foulds(
                    curr_tree,
                    attr_t1="allele_events_list_str",
                    attr_t2="allele_events_list_str",
                    expand_polytomies=False,
                    unrooted_trees=False)
            logging.info("curr tree distance root %d, unroot %d, pen_ll %f", root_rf_res[0], unroot_rf_res[0], curr_pen_ll)

        logging.info("final tree distance, root %d, unroot %d, pen_ll %f", root_rf_res[0], unroot_rf_res[0], curr_pen_ll)
        logging.info(curr_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

if __name__ == "__main__":
    main()
