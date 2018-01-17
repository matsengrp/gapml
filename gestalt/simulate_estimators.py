"""
A simulation engine to see how well cell lineage estimation performs
Right now, only contains parsimony
"""

from __future__ import division, print_function
import pickle
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')

from cell_state import CellState, CellTypeTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from barcode import Barcode
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
        nargs='+',
        default=[0.1 for _ in range(10)],
        help='target cut rates')
    parser.add_argument(
        '--repair-lambdas',
        type=float,
        default=[1, 2],
        help='repair poisson rate')
    parser.add_argument(
        '--repair-indel-probability',
        type=float,
        default=0.5,
        help='probability of deletion/insertion during repair')
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
        default=0.2,
        help='poisson parameter for distribution of insertion in cut site(s)')
    parser.add_argument(
        '--birth-lambda', type=float, default=1.25, help='birth rate')
    parser.add_argument(
        '--death-lambda', type=float, default=0.01, help='death rate')
    parser.add_argument(
        '--time', type=float, default=2, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.5,
        help='proportion cells sampled/barcodes successfully sequenced')
    parser.add_argument(
        '--n-trees', type=int, default=1, help='number of trees in forest')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--align', action='store_true')
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    # Create a cell-type tree
    cell_types = ["brain", "eye"]
    cell_type_tree = CellTypeTree(cell_type=None, rate=0)
    cell_type_tree.add_child(
        CellTypeTree(cell_type=0, rate=0.05))
    cell_type_tree.add_child(
        CellTypeTree(cell_type=1, rate=0.05))

    # Instantiate all the simulators
    bcode_simulator = BarcodeSimulator(
        np.array(args.target_lambdas),
        np.array(args.repair_lambdas), args.repair_indel_probability,
        args.repair_deletion_lambda, args.repair_deletion_lambda,
        args.repair_insertion_lambda)
    cell_type_simulator = CellTypeSimulator(cell_type_tree)
    clt_simulator = CLTSimulator(
            args.birth_lambda,
            args.death_lambda,
            cell_type_simulator,
            bcode_simulator)

    # Simulate the trees
    forest = []
    for t in range(args.n_trees):
        clt = clt_simulator.simulate(
                Barcode(),
                CellState(categorical=cell_type_tree),
                args.time)
        forest.append(clt)

    #savefig(forest, args.outbase)

    # Now sample the leaves and see what happens with parsimony
    observer = CLTObserver(args.sampling_rate)
    for clt in forest:
        obs_leaves, pruned_clt = observer.observe_leaves(clt, seed = args.seed)
        # Let the two methods compare just in terms of topology
        # To do that, we need to collapse our tree.
        # We collapse branches if the barcodes are identical.
        for node in pruned_clt.get_descendants(strategy='postorder'):
            if str(node.up.barcode) == str(node.barcode):
                node.dist = 0
        true_tree = CollapsedTree.collapse(pruned_clt)

        # trying out with true tree!!!
        print(pruned_clt.get_ascii(attributes=["name"], show_internal=True))
        bcode_meta = BarcodeMetadata()
        model_params = CLTLikelihoodModel(pruned_clt, bcode_meta)
        approximator = ApproximatorLB(2,1)
        lasso_est = CLTLassoEstimator(0, model_params, approximator)
        lasso_est.get_likelihood(model_params)


if __name__ == "__main__":
    main()
