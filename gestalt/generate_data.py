"""
A simulation engine to create cell lineage tree and data samples
"""
from __future__ import division, print_function
import os
import sys
import csv
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import logging
import pickle
from pathlib import Path
import six

from cell_state import CellState, CellTypeTree
from cell_lineage_tree import CellLineageTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulatorBifurcating
from clt_simulator_simple import CLTSimulatorOneLayer, CLTSimulatorTwoLayers
from clt_likelihood_model import CLTLikelihoodModel
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver, ObservedAlignedSeq
from alignment import AlignerNW
from barcode_metadata import BarcodeMetadata

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
        default=[0.02] * 10,
        help='target cut rates -- will get slightly perturbed for the true value')
    parser.add_argument(
        '--variance-target-lambdas',
        type=float,
        default=0.0008,
        help='variance of target cut rates (so variance of perturbations)')
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
        '--birth-lambda', type=float, default=1.8, help='birth rate')
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
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=10)
    parser.add_argument('--max-clt-nodes', type=int, default=40000)

    parser.set_defaults(use_cell_state=False)
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/generate_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.out_obs_data = "%s/obs_data.pkl" % args.out_folder
    args.out_true_model = "%s/true_model.pkl" % args.out_folder

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
            logging.info("ValueError warning.... %s", str(e))
            continue
        except AssertionError as e:
            logging.info("AssertionError warning ... %s", str(e))
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

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(seed=args.model_seed)

    barcode_orig = BarcodeMetadata.create_fake_barcode_str(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
    bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig, num_barcodes = args.num_barcodes)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    perturbations = np.random.uniform(size=args.num_targets) - 0.5
    perturbations = perturbations / np.sqrt(np.var(perturbations)) * np.sqrt(args.variance_target_lambdas)
    args.target_lambdas = np.array(args.target_lambdas) + perturbations
    min_lambda = np.min(args.target_lambdas)
    if min_lambda < 0:
        boost = 0.00001
        args.target_lambdas = args.target_lambdas - min_lambda + boost
        args.birth_lambda += -min_lambda + boost
        args.death_lambda += -min_lambda + boost
    assert np.isclose(np.var(args.target_lambdas), args.variance_target_lambdas)
    logging.info("args.target_lambdas %s" % str(args.target_lambdas))

    # Create a cell-type tree
    cell_type_tree = create_cell_type_tree(args)

    logging.info(str(args))

    sess = tf.InteractiveSession()
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
    clt_model.tot_time = args.time
    tf.global_variables_initializer().run()

    _, obs_leaves, true_tree = create_cell_lineage_tree(
            args,
            clt_model,
            bifurcating_only=True)

    true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)
    leaf_strs = [l.allele_events_list_str for l in true_tree]
    assert len(leaf_strs) == len(set(leaf_strs))

    for leaf in true_tree:
        assert np.isclose(leaf.get_distance(true_tree), args.time)

    # Print fun facts about the data
    num_leaves = len(true_tree)
    logging.info("True tree topology, num leaves %d" % num_leaves)
    logging.info(true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(true_tree.get_ascii(attributes=["cell_state"], show_internal=True))
    logging.info(true_tree.get_ascii(attributes=["observed"], show_internal=True))
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))
    # Get parsimony score of true tree
    pars_score = true_tree.get_parsimony_score()
    logging.info("Oracle tree parsimony score %d", pars_score)

    # Save the observed data
    with open(args.out_obs_data, "wb") as f:
        out_dict = {
            "bcode_meta": bcode_meta,
            "obs_leaves": obs_leaves,
        }
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

    # Save the true data
    with open(args.out_true_model, "wb") as f:
        out_dict = {
            "true_model_params": clt_model.get_vars_as_dict(),
            "true_tree": true_tree,
            "bcode_meta": bcode_meta,
            "args": args}
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
