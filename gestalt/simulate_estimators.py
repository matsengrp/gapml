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

from cell_state import CellState, CellTypeTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulator
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator
from clt_likelihood_estimator import *
from collapsed_tree import CollapsedTree
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
        default=[0.1] * 10,
        help='target cut rates')
    parser.add_argument(
        '--repair-long-probability',
        type=float,
        nargs=2,
        default=[0.05] * 2,
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
        '--birth-lambda', type=float, default=1.25, help='birth rate')
    parser.add_argument(
        '--death-lambda', type=float, default=0.01, help='death rate')
    parser.add_argument(
        '--time', type=float, default=1.2, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.5,
        help='proportion cells sampled/alleles successfully sequenced')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lasso-param', type=float, default=0)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--align', action='store_true')
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    args.target_lambdas = np.array(args.target_lambdas)
    print("args.target_lambdas", args.target_lambdas)

    sess = tf.Session()
    with sess.as_default():
        # Create a cell-type tree
        cell_type_tree = CellTypeTree(cell_type=None, rate=0)
        cell_type_tree.add_child(
            CellTypeTree(cell_type=0, rate=0.05))
        cell_type_tree.add_child(
            CellTypeTree(cell_type=1, rate=0.05))

        # Instantiate all the simulators
        bcode_meta = BarcodeMetadata()
        model_params = CLTLikelihoodModel(
                None,
                bcode_meta,
                sess,
                target_lams = np.array(args.target_lambdas),
                trim_long_probs = np.array(args.repair_long_probability),
                trim_zero_prob = args.repair_indel_probability,
                trim_poissons = np.array([args.repair_deletion_lambda, args.repair_deletion_lambda]),
                insert_zero_prob = args.repair_indel_probability,
                insert_poisson = args.repair_insertion_lambda)
        tf.global_variables_initializer().run()

        allele_simulator = AlleleSimulatorSimultaneous(
            bcode_meta,
            model_params)

        cell_type_simulator = CellTypeSimulator(cell_type_tree)
        clt_simulator = CLTSimulator(
                args.birth_lambda,
                args.death_lambda,
                cell_type_simulator,
                allele_simulator)

        # Simulate the tree
        clt = clt_simulator.simulate(
                Allele(BARCODE_V7, bcode_meta),
                CellState(categorical=cell_type_tree),
                args.time)

        # Now sample the leaves and see what happens with parsimony
        observer = CLTObserver(args.sampling_rate)
        obs_leaves, pruned_clt = observer.observe_leaves(clt, seed = args.seed)
        print("NUM LEAVES", len(pruned_clt))
        # Let the two methods compare just in terms of topology
        # To do that, we need to collapse our tree.
        true_tree = CollapsedTree.collapse(pruned_clt, deduplicate_sisters=True)

        # trying out with true tree!!!
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

        branch_lens = []
        for node in true_tree.traverse(model_params.NODE_ORDER):
            branch_lens.append(node.dist)
        print(true_tree.get_ascii(attributes=["allele_events"], show_internal=True))

        my_model = CLTLikelihoodModel(
                true_tree,
                bcode_meta,
                sess,
                # Fix the first branch length to get the scaling correct?
                branch_lens = np.concatenate([[branch_lens[0]], np.ones(len(branch_lens) - 1) * 0.5]),
                target_lams = np.ones(len(args.target_lambdas)) * 0.1 + np.random.uniform(size=len(args.target_lambdas)) * 0.1,
                trim_long_probs = np.ones(2) * 0.01,
                trim_zero_prob = 0.02,
                trim_poissons = np.ones(2),
                insert_zero_prob = 0.02,
                insert_poisson = 1.0)
        lasso_est = CLTLassoEstimator(my_model, approximator)
        lasso_est.fit(args.lasso_param, args.max_iters)
        print("---- TRUTH -----")
        print(args.target_lambdas)
        print(args.repair_long_probability)
        print(args.repair_indel_probability)
        print([args.repair_deletion_lambda, args.repair_deletion_lambda])
        print(args.repair_indel_probability)
        print(args.repair_insertion_lambda)
        print(branch_lens)
        print("ignore branch index (root)", my_model.root_node_id)

if __name__ == "__main__":
    main()
