"""
Share code for simulations
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
from tensorflow import Session
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import pickle
from pathlib import Path
from numpy import ndarray
import copy
import typing
from typing import List

from cell_lineage_tree import CellLineageTree
from cell_state import CellState, CellTypeTree
from allele import Allele
from clt_observer import CLTObserver, ObservedAlignedSeq
from clt_estimator import CLTParsimonyEstimator
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_estimator import *
from alignment import AlignerNW
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_distance import TreeDistanceMeasurer, TreeDistanceMeasurerAgg

from constants import *
from common import *
from summary_util import *

def get_parsimony_trees(
        obs_leaves: List[ObservedAlignedSeq],
        args,
        bcode_meta: BarcodeMetadata,
        do_collapse: bool=False):
    parsimony_estimator = CLTParsimonyEstimator(
            bcode_meta,
            args.out_folder,
            args.mix_path)
    #TODO: DOESN'T USE CELL STATE
    parsimony_trees = parsimony_estimator.estimate(
            obs_leaves,
            num_mix_runs=args.num_jumbles,
            do_collapse=do_collapse)
    logging.info("Total parsimony trees %d", len(parsimony_trees))

    parsimony_score = parsimony_trees[0].get_parsimony_score()
    logging.info("parsimony scores %d", parsimony_score)
    return parsimony_trees

def compare_lengths(length_dict1, length_dict2, subset, branch_plot_file, label):
    """
    Compares branch lengths, logs the results as well as plots them

    @param subset: the subset of keys in the length dicts to use for the comparison
    @param branch_plot_file: name of file to save the scatter plot to
    @param label: the label for logging/plotting
    """
    length_list1 = []
    length_list2 = []
    for k in subset:
        length_list1.append(length_dict1[k])
        length_list2.append(length_dict2[k])
    logging.info("Compare lengths %s", label)
    logging.info("pearson %s %s", label, pearsonr(length_list1, length_list2))
    logging.info("pearson (log) %s %s", label, pearsonr(np.log(length_list1), np.log(length_list2)))
    logging.info("spearman %s %s", label, spearmanr(length_list1, length_list2))
    logging.info(length_list1)
    logging.info(length_list2)
    plt.scatter(np.log(length_list1), np.log(length_list2))
    plt.savefig(branch_plot_file)

def fit_pen_likelihood(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        cell_type_tree: CellTypeTree, # set to none if not using cell type tre
        know_cell_lams: bool,
        target_lams: ndarray, # set to None if it is not known
        log_barr: float,
        max_iters: int,
        approximator: ApproximatorLB,
        sess: Session,
        tot_time: float,
        warm_start: Dict[str, ndarray] = None,
        br_len_scale: float = 0.1,
        branch_len_inners: ndarray = None, # If not given, randomly initialize
        branch_len_offsets: ndarray = None, # If not given, randomly initialize
        dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    Fit the model for the given tree topology
    @param warm_start: use the given variables to initialize the model
                        If None, then start from scratch
    @param dist_measurers: if provided, passes it to the Estimator to see
                        how distance changes thru the training procedure
    """
    # TODO: didn't implement warm start for this multifurcating case
    assert warm_start is None

    num_nodes = tree.get_num_nodes()
    if branch_len_inners is None:
        branch_len_inners = np.random.rand(num_nodes) * br_len_scale
    if branch_len_offsets is None:
        branch_len_offsets = np.random.rand(num_nodes) * br_len_scale
    target_lams_known = target_lams is not None
    if not target_lams_known:
        target_lams = 0.04 * np.ones(bcode_meta.n_targets) + np.random.uniform(size=bcode_meta.n_targets) * 0.02

    tree.label_node_ids()
    res_model = CLTLikelihoodModel(
        tree,
        bcode_meta,
        sess,
        target_lams = target_lams,
        target_lams_known=target_lams_known,
        branch_len_inners = branch_len_inners,
        branch_len_offsets = branch_len_offsets,
        cell_type_tree = cell_type_tree,
        cell_lambdas_known = know_cell_lams,
        tot_time = tot_time)
    estimator = CLTPenalizedEstimator(
            res_model,
            approximator,
            log_barr)

    # Initialize with parameters such that the branch lengths are positive
    res_model.initialize_branch_lens(br_len_scale=br_len_scale)

    history = estimator.fit(max_iters, dist_measurers = dist_measurers)

    return history, res_model
