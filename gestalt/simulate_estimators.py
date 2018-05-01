"""
A simulation engine to see how well cell lineage estimation performs
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
from collapsed_tree import collapse_zero_lens

from tree_distance import *
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
            '--optim-seed',
            type=int,
            default=0,
            help="Seed for generating the model")
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
    parser.add_argument('--max-iters', type=int, default=2)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=10)
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
    parser.add_argument('--use-parsimony', action='store_true', help="use mix (CS parsimony) to estimate tree topologies")
    parser.add_argument('--topology-only', action='store_true', help="topology only")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    args.scratch_dir = os.path.join(args.out_folder, "scratch")
    args.fitted_models_file = "%s/fitted.pkl" % args.out_folder
    args.branch_plot_file = "%s/branch_lens.png" % args.out_folder

    if args.use_parsimony and args.use_cell_state:
        raise ValueError("Cannot use parsimony while observing cell state...")

    if args.use_parsimony:
        # check that there is no infile in the current folder -- this will
        # screw up mix because it will use the wrong input file
        my_file = Path("infile")
        assert not my_file.exists()

    return args

def set_branch_lens(tree: CellLineageTree, br_len_vec: ndarray):
    """
    Set the branch lengths in this tree according to br_len_vec
    @param br_len_vec: each element corresponds to a node in the tree, per the node in preorder
    """
    tree.label_node_ids()
    for n in tree.traverse():
        if n.is_root():
            n.dist = 0
        else:
            n.dist = br_len_vec[n.node_id]

    for leaf in tree:
        print("leaf to root dist", leaf.get_distance(tree))

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

        clt, obs_leaves, true_tree = create_cell_lineage_tree(args, clt_model)
        # Gather true branch lengths
        true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)
        true_branch_lens = {}
        for node in true_tree.traverse(CLTLikelihoodModel.NODE_ORDER):
            if not node.is_root():
                true_branch_lens[node.node_id] = node.dist

        # Get parsimony score of tree?
        pars_score = true_tree.get_parsimony_score()
        logging.info("Oracle tree parsimony score %d", pars_score)

        # Save the data
        save_model_data(
                args.model_data_file,
                clt_model.get_vars_as_dict(),
                cell_type_tree,
                obs_leaves,
                true_tree,
                clt)
        # Print fun facts about the data
        logging.info("Full clt leaves %d" % len(clt))
        logging.info("True tree topology, num leaves %d", len(true_tree))
        logging.info(true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        logging.info(true_tree.get_ascii(attributes=["cell_state"], show_internal=True))
        logging.info(true_tree.get_ascii(attributes=["observed"], show_internal=True))
        logging.info("Number of uniq obs alleles %d", len(obs_leaves))

        # Get the parsimony-estimated topologies
        parsimony_trees = get_parsimony_trees(
                obs_leaves,
                args,
                bcode_meta,
                true_tree,
                args.max_trees) if args.use_parsimony else {}
        if args.topology_only:
            print("Done! You only wanted topology estimation")
            return

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

        # Fit parsimony trees -- only look at a couple trees per RF distance
        #fitting_results = {}
        #for rf_dist, pars_trees in parsimony_tree_dict.items():
        #    fitting_results[rf_dist] = []
        #    logging.info(
        #            "There are %d trees with RF %d",
        #            len(pars_trees),
        #            rf_dist)
        #    for tree in pars_trees:
        #        pen_log_lik, _, res_model = fit_pen_likelihood(
        #                tree,
        #                bcode_meta,
        #                cell_type_tree if args.use_cell_state else None,
        #                args.know_cell_lambdas,
        #                np.array(args.target_lambdas) if args.know_target_lambdas else None,
        #                args.log_barr,
        #                args.max_iters,
        #                approximator,
        #                sess)
        #        fitting_results[rf_dist].append((
        #            pen_log_lik,
        #            res_model))

        #        # Print some summaries
        #        logging.info("Mix pen log lik %f RF %d", pen_log_lik, rf_dist)

        ## Correlation between RF dist and likelihood among parsimony trees
        #if fitting_results:
        #    rf_dists = []
        #    pen_log_liks = []
        #    for rf_dist, res in fitting_results.items():
        #        for r in res:
        #            rf_dists.append(rf_dist)
        #            pen_log_liks.append(r[0][0])
        #    logging.info("rf_dists %s", str(rf_dists))
        #    logging.info("pen log liks %s", str(pen_log_liks))
        #    logging.info("pearson rf to log lik %s", pearsonr(rf_dists, pen_log_liks))
        #    logging.info("spearman rf to log lik %s", spearmanr(rf_dists, pen_log_liks))
        #    plt.scatter(rf_dists, pen_log_liks)
        #    plt.savefig("%s/rf_dist_to_ll.png" % args.out_folder)

        # Fit oracle tree
        coll_tree = true_tree.copy("deepcopy")
        for n in coll_tree.traverse():
            n.name = n.allele_events_list_str
            if not n.is_root():
                if n.allele_events_list_str == n.up.allele_events_list_str:
                    n.dist = 0
        coll_tree = collapse_zero_lens(coll_tree)
        print(coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        print(coll_tree.get_ascii(attributes=["observed"], show_internal=True))

        np.random.seed(seed=args.optim_seed)
        pen_log_lik, _, oracle_model = fit_pen_likelihood(
                coll_tree,
                bcode_meta,
                cell_type_tree if args.use_cell_state else None,
                args.know_cell_lambdas,
                np.array(args.target_lambdas) if args.know_target_lambdas else None,
                args.log_barr,
                args.max_iters,
                approximator,
                sess)

        br_lens = oracle_model.get_branch_lens()
        print("br lens", br_lens)
        br_len_offsets = oracle_model.get_vars_as_dict()["branch_len_offsets"]
        print("br len off", br_len_offsets)
        #set_branch_lens(coll_tree, br_lens + br_len_offsets)
        #print(coll_tree.get_ascii(attributes=["dist"], show_internal=True))
        #fitting_results["oracle"] = [(pen_log_lik, oracle_model)]
        #save_fitted_models(args.fitted_models_file, fitting_results)

        bifurc_tree = oracle_model.get_fitted_bifurcating_tree()
        print(bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
        print(bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        tree_dist_measurers = TreeDistanceMeasurerAgg([
                UnrootRFDistanceMeasurer,
                RootRFDistanceMeasurer,
                SPRDistanceMeasurer,
                MRCADistanceMeasurer],
                true_tree,
                args.scratch_dir)
        final_dist_dicts = tree_dist_measurers.get_tree_dists([bifurc_tree])
        print(final_dist_dicts)



        #print("True tree score %f", pen_log_lik)

        logging.info("---- ORACLE -----")
        for v in oracle_model.get_vars():
            logging.info(v)
        logging.info("---- TRUTH -----")
        logging.info(args.target_lambdas)
        logging.info(args.repair_long_probability)
        logging.info(args.repair_indel_probability)
        logging.info([args.repair_deletion_lambda, args.repair_deletion_lambda])
        logging.info(args.repair_indel_probability)
        logging.info(args.repair_insertion_lambda)
        logging.info(args.cell_rates)

        ## Compare branch lengths
        #subset = [
        #        node.node_id for node in true_tree.traverse()
        #        if not node.is_leaf() and not node.is_root()]
        #est_branch_lens = oracle_model.get_branch_lens()
        #compare_lengths(
        #        true_branch_lens,
        #        est_branch_lens,
        #        subset,
        #        branch_plot_file=args.branch_plot_file,
        #        label="oracle est vs true branches")

        ## Also compare target estimates
        #fitted_vars = oracle_model.get_vars()
        #logging.info("pearson target %s", pearsonr(args.target_lambdas, fitted_vars[0]))

if __name__ == "__main__":
    main()
