"""
A simulation engine to see how well cell lineage estimation performs
"""
from __future__ import division, print_function
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import pickle

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

from constants import *
from common import *
from summary_util import *

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
        '--birth-lambda', type=float, default=1, help='birth rate')
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
            '--ridge-param',
            type=float,
            default=0.2,
            help="ridge parameter on the branch lengths")
    parser.add_argument(
            '--lasso-param',
            type=float,
            default=0.1,
            help="lasso parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=100)
    parser.add_argument('--max-clt-nodes', type=int, default=8000)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--use-cell-state', action='store_true')
    parser.add_argument('--use-parsimony', action='store_true', help="use mix (CS parsimony) to estimate tree topologies")
    parser.add_argument('--topology-only', action='store_true', help="topology only")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    args.fitted_models_file = "%s/fitted.pkl" % args.out_folder

    if args.use_parsimony and args.use_cell_state:
        raise ValueError("Cannot use parsimony while observing cell state...")

    return args

def create_cell_type_tree(args):
    # This first rate means nothing!
    cell_type_tree = CellTypeTree(cell_type=0, rate=0.1)
    args.cell_rates = [0.20, 0.25, 0.15, 0.15]
    cell1 = CellTypeTree(cell_type=1, rate=args.cell_rates[0])
    cell2 = CellTypeTree(cell_type=2, rate=args.cell_rates[1])
    cell3 = CellTypeTree(cell_type=3, rate=args.cell_rates[2])
    cell4 = CellTypeTree(cell_type=4, rate=args.cell_rates[3])
    cell_type_tree.add_child(cell1)
    cell_type_tree.add_child(cell2)
    cell2.add_child(cell3)
    cell1.add_child(cell4)
    return cell_type_tree

def create_simulators(args, clt_model):
    allele_simulator = AlleleSimulatorSimultaneous(clt_model)
    # TODO: merge cell type simulator into allele simulator
    cell_type_simulator = CellTypeSimulator(clt_model.cell_type_tree)
    if args.single_layer:
        clt_simulator = CLTSimulatorOneLayer(
                cell_type_simulator,
                allele_simulator)
    elif args.two_layers:
        clt_simulator = CLTSimulatorTwoLayers(
                cell_type_simulator,
                allele_simulator)
    else:
        clt_simulator = CLTSimulatorBifurcating(
                args.birth_lambda,
                args.death_lambda,
                cell_type_simulator,
                allele_simulator)
    observer = CLTObserver()
    return clt_simulator, observer

def create_cell_lineage_tree(args, clt_model):
    clt_simulator, observer = create_simulators(args, clt_model)

    # Keep trying to make CLT until enough leaves in observed tree
    obs_leaves = set()
    MAX_TRIES = 10
    num_tries = 0
    clt = clt_simulator.simulate(
            tree_seed = args.model_seed,
            data_seed = args.data_seed,
            time = args.time,
            max_nodes = args.max_clt_nodes)
    sampling_rate = args.sampling_rate
    while (len(obs_leaves) < args.min_leaves or len(obs_leaves) >= args.max_leaves) and sampling_rate <= 1:
        # Now sample the leaves and create the true topology
        obs_leaves, true_tree = observer.observe_leaves(
                sampling_rate,
                clt,
                seed=args.model_seed,
                observe_cell_state=args.use_cell_state)
        logging.info("sampling rate %f, num leaves %d", sampling_rate, len(obs_leaves))
        num_tries += 1
        if len(obs_leaves) < args.min_leaves:
            sampling_rate += 0.025
        elif len(obs_leaves) >= args.max_leaves:
            sampling_rate = max(1e-3, sampling_rate - 0.05)

    if len(obs_leaves) < args.min_leaves:
        raise Exception("Could not manage to get enough leaves")
    true_tree.label_tree_with_strs()

    # Check all leaves unique because rf distance code requires leaves to be unique
    # The only reason leaves wouldn't be unique is if we are observing cell state OR
    # we happen to have the same allele arise spontaneously in different parts of the tree.
    if not args.use_cell_state:
        assert len(set([n.allele_events_list_str for n in true_tree])) == len(true_tree), "leaves must be unique"

    return clt, obs_leaves, true_tree

def get_parsimony_trees(obs_leaves, args, bcode_meta, true_tree, max_uniq_trees=100):
    parsimony_estimator = CLTParsimonyEstimator(
            bcode_meta,
            args.out_folder,
            args.mix_path)
    #TODO: DOESN'T USE CELL STATE
    parsimony_trees = parsimony_estimator.estimate(
            obs_leaves,
            max_uniq_trees=max_uniq_trees)
    logging.info("Total parsimony trees %d", len(parsimony_trees))

    # Sort the parsimony trees into their robinson foulds distance from the truth
    parsimony_tree_dict = {}
    for tree in parsimony_trees:
        rf_dist_res = true_tree.robinson_foulds(
                tree,
                attr_t1="allele_events_list_str",
                attr_t2="allele_events_list_str",
                # Not only is this option buggy, I think we actually want the unrooted
                # calculation -- we want to compare if the MRCAs of the two trees
                # are the same. This means partitioning along internal edges and
                # doing a symmetric difference of the sets of partitions.
                expand_polytomies=False,
                unrooted_trees=True)
        rf_dist_max = rf_dist_res[1]
        rf_dist = rf_dist_res[0]
        logging.info("rf dist %d (max %d)", rf_dist, rf_dist_max)
        logging.info(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        logging.info(tree.get_ascii(attributes=["observed"], show_internal=True))
        if rf_dist not in parsimony_tree_dict:
            parsimony_tree_dict[rf_dist] = [tree]
        else:
            parsimony_tree_dict[rf_dist].append(tree)
    return parsimony_tree_dict

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
        tf.global_variables_initializer().run()

        clt, obs_leaves, true_tree = create_cell_lineage_tree(args, clt_model)

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
        parsimony_tree_dict = get_parsimony_trees(
                obs_leaves,
                args,
                bcode_meta,
                true_tree) if args.use_parsimony else {}
        if args.topology_only:
            print("Done! You only wanted topology estimation")
            return

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)
        def fit_pen_likelihood(tree):
            num_nodes = len([t for t in tree.traverse()])

            if args.know_target_lambdas:
                target_lams = np.array(args.target_lambdas)
            else:
                target_lams = 0.3 * np.ones(args.target_lambdas.size) + np.random.uniform(size=args.num_targets) * 0.08

            res_model = CLTLikelihoodModel(
                    tree,
                    bcode_meta,
                    sess,
                    target_lams = target_lams,
                    target_lams_known=args.know_target_lambdas,
                    #trim_long_probs = np.array(args.repair_long_probability),
                #trim_zero_prob = args.repair_indel_probability,
                #trim_poissons = np.array([args.repair_deletion_lambda, args.repair_deletion_lambda]),
                #insert_zero_prob = args.repair_indel_probability,
                #insert_poisson = args.repair_insertion_lambda,
                group_branch_lens = np.ones(num_nodes) * 0.3,
                group_branch_lens_known = args.const_branch_len,
                #branch_len_perturbs = np.array([n.dist for n in tree.traverse("preorder")]) - 0.3,
                branch_len_perturbs = np.random.randn(num_nodes) * 0.05 if not args.const_branch_len else np.zeros(num_nodes),
                cell_type_tree = cell_type_tree if args.use_cell_state else None,
                cell_lambdas_known = args.know_cell_lambdas)
            estimator = CLTPenalizedEstimator(
                    res_model,
                    approximator,
                    args.ridge_param,
                    args.lasso_param)
            pen_log_lik = estimator.fit(
                    args.num_inits,
                    args.max_iters)
            return pen_log_lik, res_model

        # Fit parsimony trees -- only look at a couple trees per RF distance
        fitting_results = {}
        for rf_dist, pars_trees in parsimony_tree_dict.items():
            fitting_results[rf_dist] = []
            logging.info("There are %d trees with RF %d", len(pars_trees), rf_dist)
            for tree in pars_trees[:2]:
                pen_log_lik, res_model = fit_pen_likelihood(tree)
                logging.info("Mix pen log lik %f RF %d", pen_log_lik, rf_dist)
                fitting_results[rf_dist].append((
                    pen_log_lik,
                    res_model))

        # Fit oracle tree
        pen_log_lik, oracle_model = fit_pen_likelihood(true_tree)
        fitting_results["oracle"] = [(pen_log_lik, oracle_model)]
        save_fitted_models(args.fitted_models_file, fitting_results)
        logging.info("True tree score %f", pen_log_lik)

        # Correlation between RF dist and likelihood
        rf_dists = []
        pen_log_liks = []
        for rf_dist, res in fitting_results.items():
            if rf_dist != "oracle":
                for r in res:
                    rf_dists.append(rf_dist)
                    pen_log_liks.append(r[0])
        logging.info("rf_dists %s", str(rf_dists))
        logging.info("pen log liks %s", str(pen_log_liks))
        logging.info("pearson rf to log lik %s", pearsonr(rf_dists, pen_log_liks))
        logging.info("spearman rf to log lik %s", spearmanr(rf_dists, pen_log_liks))

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

        fitted_vars = oracle_model.get_vars()
        est_branch_lens = oracle_model.get_branch_lens()
        # Gather true branch lengths
        true_branch_lens = {}
        for node in oracle_model.topology.traverse(oracle_model.NODE_ORDER):
            if not node.is_root():
                true_branch_lens[node.node_id] = node.dist
        logging.info("====branch lens (true, est)=====")
        true_br_list = []
        est_br_list = []
        for k in true_branch_lens.keys():
            true_br_list.append(true_branch_lens[k])
            est_br_list.append(est_branch_lens[k])
            logging.info("%f %f", true_branch_lens[k], est_branch_lens[k])
        logging.info("pearson branch %s", pearsonr(true_br_list, est_br_list))
        logging.info("spearman branch %s", spearmanr(true_br_list, est_br_list))
        logging.info("pearson target %s", pearsonr(args.target_lambdas, fitted_vars[0]))

if __name__ == "__main__":
    main()
