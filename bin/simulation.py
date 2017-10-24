#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import pickle
import sys
if sys.version_info < (3, 0):
    from string import maketrans
else:
    maketrans = str.maketrans
from collections import Counter
import numpy as np
import scipy, argparse, copy, re
from scipy.stats import expon, poisson
from numpy.random import choice, random
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_style('ticks')
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO, SeqIO
from ete3 import TreeNode, NodeStyle, TreeStyle, faces, SeqMotifFace, add_face_to_node
from collapsed_tree import CollapsedTree

from constants import BARCODE_V7


def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        'outbase', type=str, help='base name for plot and fastq output')
    parser.add_argument(
        '--target-lambdas',
        type=float,
        nargs='+',
        default=[1 for _ in range(10)],
        help='target cut poisson rates')
    parser.add_argument(
        '--repair-lambda', type=float, default=10, help='repair poisson rate')
    parser.add_argument(
        '--repair-indel-probability',
        type=float,
        default=.1,
        help='probability of deletion/insertion during repair')
    parser.add_argument(
        '--repair-deletion-lambda',
        type=float,
        default=2,
        help=
        'poisson parameter for distribution of symmetric deltion about cut site(s)'
    )
    parser.add_argument(
        '--repair-insertion-lambda',
        type=float,
        default=.5,
        help='poisson parameter for distribution of insertion in cut site(s)')
    parser.add_argument(
        '--birth-lambda', type=float, default=0.4, help='birth rate')
    parser.add_argument(
        '--death-lambda', type=float, default=0.01, help='death rate')
    parser.add_argument(
        '--time', type=float, default=5, help='how much time to simulate')
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
    indel_boundary(forest, args.outbase)
    # NOTE: function below not yet implemented
    # event_joint(forest, args.outbase)
    write_sequences(forest, args.outbase)
    render(forest, args.outbase)
    summary_plots(forest, args.outbase + '.summary_plots.pdf')

    with open(args.outbase + ".pkl", "wb") as f_pkl:
        pickle.dump(forest, f_pkl)


if __name__ == "__main__":
    main()
