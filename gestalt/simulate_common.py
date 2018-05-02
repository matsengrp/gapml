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
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulatorBifurcating
from clt_simulator_simple import CLTSimulatorOneLayer, CLTSimulatorTwoLayers
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver, ObservedAlignedSeq
from clt_estimator import CLTParsimonyEstimator
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

def create_cell_lineage_tree(args, clt_model, bifurcating_only: bool = False):
    """
    @param bifurcating_only: return a collapsed clt that is bifurcating (so sample leaves
                            so that we only get bifurcations)
    @return original clt, the set of observed leaves, and the true topology for the observed leaves
    """
    clt_simulator, observer = create_simulators(args, clt_model)

    # Keep trying to make CLT until enough leaves in observed tree
    obs_leaves = set()
    MAX_TRIES = 20
    for i in range(MAX_TRIES):
        args.model_seed += 1
        try:
            print("attempt", i)
            clt = clt_simulator.simulate(
                tree_seed = args.model_seed,
                data_seed = args.data_seed,
                time = args.time,
                max_nodes = args.max_clt_nodes)
            clt.label_node_ids()

            sampling_rate = args.sampling_rate
            while (len(obs_leaves) < args.min_leaves or len(obs_leaves) >= args.max_leaves) and sampling_rate <= 1:
                # Now sample the leaves and create the true topology
                # Keep changing the sampling rate if we can't get the right number of leaves
                obs_leaves, true_tree = observer.observe_leaves(
                        sampling_rate,
                        clt,
                        seed=args.model_seed,
                        observe_cell_state=args.use_cell_state,
                        bifurcating_only=bifurcating_only)

                logging.info("sampling rate %f, num leaves %d", sampling_rate, len(obs_leaves))
                if len(obs_leaves) < args.min_leaves:
                    sampling_rate += 0.02
                elif len(obs_leaves) >= args.max_leaves:
                    sampling_rate = max(1e-3, sampling_rate - 0.05)
                else:
                    break

            logging.info("final? sampling rate %f, num leaves %d", sampling_rate, len(obs_leaves))
            if len(obs_leaves) >= args.min_leaves and len(obs_leaves) <= args.max_leaves:
                # Done creating the tree
                break
        except ValueError as e:
            print(e)
            continue
        except AssertionError as e:
            print(e)
            continue

    if len(obs_leaves) < args.min_leaves:
        raise Exception("Could not manage to get enough leaves")

    logging.info(true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    # Check all leaves unique because rf distance code requires leaves to be unique
    # The only reason leaves wouldn't be unique is if we are observing cell state OR
    # we happen to have the same allele arise spontaneously in different parts of the tree.
    if not args.use_cell_state:
        uniq_leaves = set()
        for n in true_tree:
            if n.allele_events_list_str in uniq_leaves:
                logging.info("repeated leaf %s", n.allele_events_list_str)
                clt.label_tree_with_strs()
                logging.info(clt.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
            else:
                uniq_leaves.add(n.allele_events_list_str)
        assert len(set([n.allele_events_list_str for n in true_tree])) == len(true_tree), "leaves must be unique"

    return clt, obs_leaves, true_tree

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
