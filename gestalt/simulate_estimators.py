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

from cell_state import CellTypeTree
from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator
from clt_likelihood import *
from collapsed_tree import CollapsedTree

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
        '--time', type=float, default=4, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.5,
        help='proportion cells sampled/barcodes successfully sequenced')
    parser.add_argument(
        '--n-trees', type=int, default=1, help='number of trees in forest')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    # Create a cell-type tree
    cell_types = ["brain", "eye"]
    cell_type_tree = CellTypeTree(cell_type=None, rate=0.1, probability=1.0)
    cell_type_tree.add_child(
        CellTypeTree(cell_type=0, rate=0, probability=0.5))
    cell_type_tree.add_child(
        CellTypeTree(cell_type=1, rate=0, probability=0.5))

    # Instantiate all the simulators
    bcode_simulator = BarcodeSimulator(
        np.array(args.target_lambdas),
        np.array(args.repair_lambdas), args.repair_indel_probability,
        args.repair_deletion_lambda, args.repair_deletion_lambda,
        args.repair_insertion_lambda)
    clt_simulator = CLTSimulator(args.birth_lambda, args.death_lambda,
                                 cell_type_tree, bcode_simulator)

    # Simulate the trees
    forest = []
    for t in range(args.n_trees):
        clt = clt_simulator.simulate(args.time)
        forest.append(clt)

    #savefig(forest, args.outbase)

    # Now sample the leaves and see what happens with parsimony
    observer = CLTObserver(args.sampling_rate)
    par_estimator = CLTParsimonyEstimator()
    for clt in forest:
        obs_leaves, pruned_clt = observer.observe_leaves(clt)
        # Let the two methods compare just in terms of topology
        # To do that, we need to collapse our tree.
        # We collapse branches if the barcodes are identical.
        for node in pruned_clt.get_descendants(strategy='postorder'):
            if str(node.up.barcode) == str(node.barcode):
                node.dist = 0
        true_tree = CollapsedTree.collapse(pruned_clt)

        #par_est_trees = par_estimator.estimate(obs_leaves)

        ## Display the true tree (rename leaves for visualization ease)
        #for leaf in true_tree:
        #    leaf.name = str(leaf.barcode.get_events()) + str(leaf.cell_state)
        ##    print(leaf.up.barcode.events())
        ##    print(leaf.barcode.events())
        ##    print("==%s==" % leaf.name)
        #print("TRUTH")
        #print(true_tree)

        ## For now, we just display the first estimated tree
        #par_est_t = par_est_trees[0]
        #for leaf in par_est_t:
        #    leaf.name = str(leaf.barcode.get_events()) + str(leaf.cell_state)
        #print("ESTIMATE (1 out of %d equally parsimonious trees)" %
        #      len(par_est_trees))
        #print(par_est_t)

        # trying out with true tree!!!
        # TODO: convert observed aligned seq to barcode events!
        model_params = CLTLikelihoodModel(pruned_clt, 10)
        lasso_est = CLTLassoEstimator(obs_leaves, 0, model_params)
        lasso_est.get_likelihood(model_params)


if __name__ == "__main__":
    main()
