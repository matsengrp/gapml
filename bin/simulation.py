from __future__ import division, print_function
import pickle
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')

from cell_state import CellTypeTree
from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from cell_state import CellTypeTree, CellType

from constants import *
from summary_util import *


def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        'outbase', type=str, help='base name for plot and fastq output')
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
        '--n-trees', type=int, default=1, help='number of trees in forest')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    # Create a cell-type tree
    cell_type_tree = CellTypeTree(cell_type=None, rate=0.1, probability=1.0)
    cell_type_tree.add_child(
        CellTypeTree(cell_type=CellType.BRAIN, rate=0, probability=0.5))
    cell_type_tree.add_child(
        CellTypeTree(cell_type=CellType.EYE, rate=0, probability=0.5))

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

    # Dump summary statistics
    editing_profile(forest, args.outbase)
    # indel_boundary(forest, args.outbase)
    # NOTE: function below not yet implemented
    # event_joint(forest, args.outbase)
    write_sequences(forest, args.outbase)
    savefig(forest, args.outbase)
    summary_plots(forest, args.outbase + '.summary_plots.pdf')

    with open(args.outbase + ".pkl", "wb") as f_pkl:
        pickle.dump(forest, f_pkl)


if __name__ == "__main__":
    main()
