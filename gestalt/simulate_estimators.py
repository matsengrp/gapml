"""
A simulation engine to see how well cell lineage estimation performs
"""

from __future__ import division, print_function
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr, kendalltau

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
from summary_util import *

def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        '--outbase',
        type=str,
        default="_output/test",
        help='base name for plot and fastq output')
    parser.add_argument(
        '--target-lambdas',
        type=float,
        nargs=10,
        default=[0.01] * 10,
        help='target cut rates -- will get slightly perturbed for the true value')
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
            '--pen-param',
            type=float,
            default=0,
            help="ridge parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--use-parsimony', action='store_true', help="use mix (CS parsimony) to estimate tree topologies")
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    np.random.seed(seed=args.model_seed)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    args.target_lambdas = np.array(args.target_lambdas) + np.random.uniform(size=args.num_targets) * 0.08
    print("args.target_lambdas", args.target_lambdas)

    # TODO: make this a real parameter
    use_cell_state = False

    sess = tf.Session()
    with sess.as_default():
        # Create a cell-type tree
        cell_type_tree = CellTypeTree(cell_type=0, rate=0)
        cell_type_tree.add_child(
            CellTypeTree(cell_type=1, rate=0.18))
        cell_type_tree.add_child(
            CellTypeTree(cell_type=2, rate=0.20))

        # Instantiate all the simulators
        barcode_orig = BarcodeMetadata.create_fake_barcode(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
        bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig)
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
                cell_type_tree = cell_type_tree if use_cell_state else None)
        tf.global_variables_initializer().run()

        allele_simulator = AlleleSimulatorSimultaneous(
            bcode_meta,
            clt_model)
        # TODO: merge cell type simulator into allele simulator
        cell_type_simulator = CellTypeSimulator(cell_type_tree)
        if not args.debug:
            clt_simulator = CLTSimulatorBifurcating(
                    args.birth_lambda,
                    args.death_lambda,
                    cell_type_simulator,
                    allele_simulator)
            max_nodes = 300
        else:
            clt_simulator = CLTSimulatorSimple(
                    cell_type_simulator,
                    allele_simulator)
            max_nodes = 70
        clt = clt_simulator.simulate(
                tree_seed = args.model_seed,
                data_seed = args.data_seed,
                root_allele = Allele(barcode_orig, bcode_meta),
                root_cell_state = CellState(categorical=cell_type_tree),
                time = args.time,
                max_nodes = max_nodes)

        # Now sample the leaves and create the true topology
        observer = CLTObserver(args.sampling_rate)
        obs_leaves, true_tree = observer.observe_leaves(
                clt,
                seed=args.model_seed,
                observe_cell_state=use_cell_state)
        num_nodes = len([t for t in true_tree.traverse()])

        # Get the parsimony-estimated topologies
        parsimony_estimator = CLTParsimonyEstimator(barcode_orig, bcode_meta, args.mix_path)
        #TODO: DOESN"T ACTUALLY USE CELL STATE
        parsimony_trees = parsimony_estimator.estimate(
                obs_leaves,
                use_cell_state=use_cell_state,
                max_trees=100) if args.use_parsimony else []
        if args.use_parsimony:
            print("Total parsimony trees", len(parsimony_trees))

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)
        def fit_pen_likelihood(tree):
            #TODO: right now initializes with the correct parameters
            res_model = CLTLikelihoodModel(
                    tree,
                    bcode_meta,
                    sess,
                target_lams = np.array(args.target_lambdas),
                trim_long_probs = np.array(args.repair_long_probability),
                trim_zero_prob = args.repair_indel_probability,
                trim_poissons = np.array([args.repair_deletion_lambda, args.repair_deletion_lambda]),
                insert_zero_prob = args.repair_indel_probability,
                insert_poisson = args.repair_insertion_lambda,
                group_branch_lens = np.ones(num_nodes) * 0.3,
                branch_len_perturbs = np.random.randn(num_nodes) * 0.01,
                cell_type_tree = cell_type_tree if use_cell_state else None)
            estimator = CLTPenalizedEstimator(res_model, approximator)
            pen_log_lik = estimator.fit(
                    args.pen_param,
                    args.num_inits,
                    args.max_iters)
            return pen_log_lik, res_model

        print("True tree topology, num leaves", len(true_tree))
        print(true_tree.get_ascii(attributes=["allele_events"], show_internal=True))
        print(true_tree.get_ascii(attributes=["cell_state"], show_internal=True))
        print("Number of uniq obs alleles", len(obs_leaves))

        # Fit parsimony trees
        rf_dist_trees = {}
        for tree in parsimony_trees:
            rf_dist = true_tree.robinson_foulds(
                    tree,
                    attr_t1="allele_events",
                    attr_t2="allele_events",
                    unrooted_trees=True)[0]
            if rf_dist not in rf_dist_trees:
                print("rf dist", rf_dist)
                print(tree.get_ascii(attributes=["allele_events"], show_internal=True))
                rf_dist_trees[rf_dist] = [tree]
            else:
                rf_dist_trees[rf_dist].append(tree)

        for rf_dist, pars_trees in rf_dist_trees.items():
            for tree in pars_trees[:2]:
                pen_log_lik, res_model = fit_pen_likelihood(tree)
                print("Mix pen log lik", pen_log_lik, "RF", rf_dist)

        # Fit oracle tree
        pen_log_lik, oracle_model = fit_pen_likelihood(true_tree)
        print("True tree score", pen_log_lik)

        print("---- TRUTH -----")
        for v in oracle_model.get_vars():
            print(v)
        print(args.target_lambdas)
        print(args.repair_long_probability)
        print(args.repair_indel_probability)
        print([args.repair_deletion_lambda, args.repair_deletion_lambda])
        print(args.repair_indel_probability)
        print(args.repair_insertion_lambda)

        fitted_vars = oracle_model.get_vars()
        est_branch_lens = oracle_model.get_branch_lens()
        # Gather true branch lengths
        true_branch_lens = {}
        for node in oracle_model.topology.traverse(oracle_model.NODE_ORDER):
            if not node.is_root():
                true_branch_lens[node.node_id] = node.dist
        print("====branch lens (true, est)=====")
        true_br_list = []
        est_br_list = []
        for k in true_branch_lens.keys():
            true_br_list.append(true_branch_lens[k])
            est_br_list.append(est_branch_lens[k])
            print(true_branch_lens[k], est_branch_lens[k])
        print("pearson branch (to oracle)", pearsonr(true_br_list, est_br_list))
        print("spearman branch (to oracle)", spearmanr(true_br_list, est_br_list))
        print("pearson target (to oracle)", pearsonr(args.target_lambdas, fitted_vars[0]))

        # Compare distance to root?
        true_dist_to_root = []
        est_dist_to_root = []
        # this just assigns length 1 to each branch
        # TODO: in the future, maybe assign length by number of changes?
        stupid_dist_to_root = []
        for leaf in oracle_model.topology:
            true_dist = leaf.dist
            est_dist = est_branch_lens[leaf.node_id]
            stupid_dist = 1
            curr_node = leaf
            while not curr_node.up.is_root():
                true_dist += curr_node.up.dist
                est_dist += est_branch_lens[curr_node.up.node_id]
                stupid_dist += 1
                curr_node = curr_node.up
            true_dist_to_root.append(true_dist)
            est_dist_to_root.append(est_dist)
            stupid_dist_to_root.append(stupid_dist)
        print("pearson dist to root (to oracle)", pearsonr(true_dist_to_root, est_dist_to_root))
        print("spearman dist to root (to oracle)", spearmanr(true_dist_to_root, est_dist_to_root))
        print("kendalltau dist to root (to oracle)", kendalltau(
            np.argsort(true_dist_to_root),
            np.argsort(est_dist_to_root)))
        print("pearson dist to root (to stupid)", pearsonr(true_dist_to_root, stupid_dist_to_root))
        print("spearman dist to root (to stupid)", spearmanr(true_dist_to_root, stupid_dist_to_root))
        print("kendalltau dist to root (to stupid)", kendalltau(
            np.argsort(true_dist_to_root),
            np.argsort(stupid_dist_to_root)))

if __name__ == "__main__":
    main()
