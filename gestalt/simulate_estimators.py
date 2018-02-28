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
from clt_simulator_simple import CLTSimulatorSimple
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
        '--target-lambdas',
        type=float,
        nargs=10,
        default=[0.01] * 10,
        help='target cut rates -- will get slightly perturbed for the true value')
    parser.add_argument(
        '--target-lambdas-known',
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
    parser.add_argument('--min-leaves', type=int, default=1)
    parser.add_argument('--max-leaves', type=int, default=100)
    parser.add_argument('--max-clt-nodes', type=int, default=8000)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--use-cell-state', action='store_true')
    parser.add_argument('--use-parsimony', action='store_true', help="use mix (CS parsimony) to estimate tree topologies")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.model_data_file = "%s/model_data.pkl" % args.out_folder
    return args

def create_cell_type_tree():
    # This first rate means nothing!
    cell_type_tree = CellTypeTree(cell_type=0, rate=0.1)
    cell_type_tree.add_child(
        CellTypeTree(cell_type=1, rate=0.18))
    cell_type_tree.add_child(
        CellTypeTree(cell_type=2, rate=0.20))
    return cell_type_tree

def create_simulators(args, clt_model):
    allele_simulator = AlleleSimulatorSimultaneous(clt_model)
    # TODO: merge cell type simulator into allele simulator
    cell_type_simulator = CellTypeSimulator(clt_model.cell_type_tree)
    if not args.debug:
        clt_simulator = CLTSimulatorBifurcating(
                args.birth_lambda,
                args.death_lambda,
                cell_type_simulator,
                allele_simulator)
    else:
        clt_simulator = CLTSimulatorSimple(
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
    while (len(obs_leaves) <= args.min_leaves or len(obs_leaves) >= args.max_leaves) and sampling_rate < 1:
        # Now sample the leaves and create the true topology
        obs_leaves, true_tree = observer.observe_leaves(
                sampling_rate,
                clt,
                seed=args.model_seed,
                observe_cell_state=args.use_cell_state)
        logging.info("sampling rate %f, num leaves %d", sampling_rate, len(obs_leaves))
        num_tries += 1
        if len(obs_leaves) <= args.min_leaves:
            sampling_rate += 0.025
        elif len(obs_leaves) >= args.max_leaves:
            sampling_rate = max(1e-3, sampling_rate - 0.05)

    if len(obs_leaves) <= args.min_leaves:
        raise Exception("Could not manage to get enough leaves")
    return clt, obs_leaves, true_tree

def get_parsimony_trees(obs_leaves, args, bcode_meta, true_tree, max_uniq_trees=100):
    parsimony_estimator = CLTParsimonyEstimator(bcode_meta, args.mix_path)
    #TODO: DOESN"T ACTUALLY USE CELL STATE
    parsimony_trees = parsimony_estimator.estimate(
            obs_leaves,
            use_cell_state=args.use_cell_state,
            max_uniq_trees=max_uniq_trees)
    logging.info("Total parsimony trees %d", len(parsimony_trees))

    # Sort the parsimony trees into their robinson foulds distance from the truth
    parsimony_tree_dict = {}
    for tree in parsimony_trees:
        rf_dist = true_tree.robinson_foulds(
                tree,
                attr_t1="allele_events",
                attr_t2="allele_events",
                unrooted_trees=True)[0]
        if rf_dist not in parsimony_tree_dict:
            logging.info("rf dist %d", rf_dist)
            logging.info(tree.get_ascii(attributes=["allele_events"], show_internal=True))
            parsimony_tree_dict[rf_dist] = [tree]
        else:
            parsimony_tree_dict[rf_dist].append(tree)
    return parsimony_tree_dict

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(seed=args.model_seed)

    barcode_orig = BarcodeMetadata.create_fake_barcode(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
    bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    args.target_lambdas = np.array(args.target_lambdas) + np.random.uniform(size=args.num_targets) * 0.08
    logging.info("args.target_lambdas %s" % str(args.target_lambdas))

    # Create a cell-type tree
    cell_type_tree = create_cell_type_tree()

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

        num_nodes = len([t for t in true_tree.traverse()])

        # Get the parsimony-estimated topologies
        parsimony_tree_dict = get_parsimony_trees(
                obs_leaves,
                args,
                bcode_meta,
                true_tree) if args.use_parsimony else {}

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)
        def fit_pen_likelihood(tree):
            #TODO: right now initializes with the correct parameters
            if args.target_lambdas_known:
                target_lams = np.array(args.target_lambdas)
            else:
                target_lams = 0.3 * np.ones(args.target_lambdas.size) + np.random.uniform(size=args.num_targets) * 0.08

            res_model = CLTLikelihoodModel(
                    tree,
                    bcode_meta,
                    sess,
                    target_lams = target_lams,
                    target_lams_known=args.target_lambdas_known,
                    #trim_long_probs = np.array(args.repair_long_probability),
                #trim_zero_prob = args.repair_indel_probability,
                #trim_poissons = np.array([args.repair_deletion_lambda, args.repair_deletion_lambda]),
                #insert_zero_prob = args.repair_indel_probability,
                #insert_poisson = args.repair_insertion_lambda,
                group_branch_lens = np.ones(num_nodes) * 0.3,
                branch_len_perturbs = np.random.randn(num_nodes) * 0.05,
                cell_type_tree = cell_type_tree if args.use_cell_state else None)
            estimator = CLTPenalizedEstimator(
                    res_model,
                    approximator,
                    args.ridge_param,
                    args.lasso_param)
            pen_log_lik = estimator.fit(
                    args.num_inits,
                    args.max_iters)
            return pen_log_lik, res_model

        logging.info("True tree topology, num leaves %d", len(true_tree))
        logging.info(true_tree.get_ascii(attributes=["allele_events"], show_internal=True))
        logging.info(true_tree.get_ascii(attributes=["cell_state"], show_internal=True))
        logging.info(true_tree.get_ascii(attributes=["observed"], show_internal=True))
        logging.info("Number of uniq obs alleles %d", len(obs_leaves))

        # Fit parsimony trees -- only look at a couple trees per RF distance
        for rf_dist, pars_trees in parsimony_tree_dict.items():
            for tree in pars_trees[:2]:
                pen_log_lik, res_model = fit_pen_likelihood(tree)
                logging.info("Mix pen log lik %f RF %d", pen_log_lik, rf_dist)

        # Fit oracle tree
        pen_log_lik, oracle_model = fit_pen_likelihood(true_tree)
        logging.info("True tree score %f", pen_log_lik)

        logging.info("---- TRUTH -----")
        for v in oracle_model.get_vars():
            logging.info(v)
        logging.info(args.target_lambdas)
        logging.info(args.repair_long_probability)
        logging.info(args.repair_indel_probability)
        logging.info([args.repair_deletion_lambda, args.repair_deletion_lambda])
        logging.info(args.repair_indel_probability)
        logging.info(args.repair_insertion_lambda)

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
