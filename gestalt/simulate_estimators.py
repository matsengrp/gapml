"""
This code fits model parameters and branch lengths
constrained to the multifurcating tree topology.

Searches over bifurcating trees with the same multifurcating tree
by using a continuous parameterization.
Suppose constant events when resolving multifurcations.
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
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import pickle
from pathlib import Path
import six

from cell_state import CellState, CellTypeTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulatorBifurcating
from clt_simulator_simple import CLTSimulatorOneLayer, CLTSimulatorTwoLayers
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator
from alignment import AlignerNW
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from collapsed_tree import collapse_zero_lens

from parallel_worker import BatchSubmissionManager
from likelihood_scorer import LikelihoodScorer
from tree_distance import *
from constants import *
from common import *
from summary_util import *
from simulate_common import *

def parse_args():
    parser = argparse.ArgumentParser(description='fit topology and branch lengths for GESTALT')
    parser.add_argument(
        '--obs-data',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--tree-topology',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help='pkl file with tree topology')
    parser.add_argument(
        '--true-model',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--out-folder',
        type=str,
        default="_output",
        help='folder to put output in')
    parser.add_argument(
        '--know-target-lambdas',
        action='store_true')
    parser.add_argument(
        '--know-cell-lambdas',
        action='store_true')
    parser.add_argument(
        '--const-branch-len',
        action='store_true')
    parser.add_argument(
        '--time', type=float, default=1.2, help='how much time to fit for')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument(
            '--seed',
            type=int,
            default=40,
            help="Seed for generating the model")
    parser.add_argument(
            '--log-barr',
            type=float,
            default=0.001,
            help="log barrier parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--do-distributed', action='store_true')

    parser.set_defaults(use_cell_state=False)
    args = parser.parse_args()
    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/fit_log.txt" % args.out_folder
    print("Log file", args.log_file)
    args.csv_out = "%s/estimators_multifurc.csv" % args.out_folder
    args.pickle_out = "%s/estimators_multifurc.pkl" % args.out_folder
    args.scratch_dir = os.path.join(args.out_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(seed=args.seed)

    with open(args.obs_data, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))

    true_model_dict = None
    with open(args.true_model, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)
        oracle_dist_measurers = TreeDistanceMeasurerAgg([
            UnrootRFDistanceMeasurer,
            RootRFDistanceMeasurer,
            SPRDistanceMeasurer,
            MRCADistanceMeasurer],
            true_model_dict["true_tree"],
            args.scratch_dir)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Instantiate approximator used by our penalized MLE
    approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

    def _make_likelihood_scorer(tree, name):
        return LikelihoodScorer(
                args.optim_seed,
                tree,
                bcode_meta,
                None, # Do not use cell type info
                args.know_cell_lambdas,
                np.array(args.target_lambdas) if args.know_target_lambdas else None,
                args.log_barr,
                args.max_iters,
                approximator,
                tot_time = args.time,
                dist_measurers = tree_dist_measurers,
                # Send in name of the tree as auxiliary information
                aux = name)

    # The dict of trees to fit models for
    trees_to_test = {}

    # Get the parsimony-estimated topologies
    if args.use_parsimony:
        parsimony_trees = get_parsimony_trees(
            obs_leaves,
            args,
            bcode_meta,
            do_collapse=False)

        oracle_measurer = UnrootRFDistanceMeasurer(true_tree, args.scratch_dir)

        min_dist = oracle_measurer.get_dist(parsimony_trees[0])
        best_parsimony_tree = parsimony_trees[0]
        parsimony_dists = [min_dist]
        for pars_tree in parsimony_trees:
            tree_dist = oracle_measurer.get_dist(pars_tree)
            parsimony_dists.append(tree_dist)
            if tree_dist < min_dist:
                best_parsimony_tree = pars_tree
                min_dist = tree_dist
        logging.info("Uniq parsimony %s distances: %s", oracle_measurer.name, np.unique(parsimony_dists))
        logging.info("Mean parsimony %s distance: %f", oracle_measurer.name, np.mean(parsimony_dists))
        logging.info("Min parsimony %s distance: %d", oracle_measurer.name, min_dist)

        # Add the closest tree from parsimony to the list
        trees_to_test["best_parsimony"] = best_parsimony_tree
        logging.info("Best parsimony -- not collapsed")
        logging.info(best_parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        # Add the collapsed version of the closest tree from parsimony to the list
        best_coll_tree = collapse_internally_labelled_tree(best_parsimony_tree)
        trees_to_test["best_pars_multifurc"] = best_coll_tree
        logging.info("Best parsimony -- collapsed")
        logging.info(best_coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        # Add a random tree from parsimony to the list
        random_parsimony_tree = parsimony_trees[np.random.randint(low=0, high=len(parsimony_trees))]
        trees_to_test["random_parsimony"] = random_parsimony_tree
        logging.info("Random parsimony -- not collapsed")
        logging.info(random_parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        # Add a collapsed version of the random tree from parsimony to the list
        random_coll_tree = collapse_internally_labelled_tree(random_parsimony_tree)
        rf_measurer = UnrootRFDistanceMeasurer(random_coll_tree, None)
        rf_dist = rf_measurer.get_dist(best_coll_tree)
        logging.info("Random parsimony collapse rf_dist from best parsimony collapse: %d", rf_dist)
        if rf_dist > 0:
            trees_to_test["random_multifurc"] = random_coll_tree
            logging.info("Random parsimony -- collapsed")
            logging.info(random_coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    #######
    # Fit continuous parameterization of ambiguous multifurcating trees
    ######
    coll_parsimony_trees = get_parsimony_trees(
        obs_leaves,
        args,
        bcode_meta,
        do_collapse=True)
    logging.info("Number of collapsed parsimony trees %d", len(coll_parsimony_trees))

    # Add a collapsed version of the oracle tree from parsimony to the list
    oracle_coll_tree = collapse_internally_labelled_tree(true_tree)
    trees_to_test["oracle_multifurc"] = oracle_coll_tree
    logging.info("Oracle collapsed tree")
    logging.info(oracle_coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    # Make workers and run workers
    worker_list = []
    for key, tree in trees_to_test.items():
        worker_list.append(_make_likelihood_scorer(tree, key))
    if args.do_distributed and len(worker_list) > 1:
        # Submit jobs to slurm
        batch_manager = BatchSubmissionManager(
                worker_list=worker_list,
                shared_obj=None,
                # Each tree is its separate slurm job
                num_approx_batches=len(worker_list),
                worker_folder=args.scratch_dir)
        successful_res_workers = batch_manager.run(successful_only=True)
    else:
        # Run jobs locally
        successful_res_workers = [(worker.do_work_directly(sess), worker) for worker in worker_list]

    # Process workers
    all_print_results = []
    for res, worker in successful_res_workers:
        result_print_list = tree_dist_measurers.get_tree_dists([res.fitted_bifurc_tree])[0]
        pearson_target = pearsonr(args.target_lambdas, res.model_params_dict["target_lams"])
        result_print_list["pearson_target_corr"] = pearson_target[0]
        result_print_list["pearson_target_pval"] = pearson_target[1]
        result_print_list["target_lam_dist"] = np.linalg.norm(args.target_lambdas - res.model_params_dict["target_lams"])
        result_print_list["name"] = worker.aux
        result_print_list["num_leaves"] = num_leaves
        result_print_list["model_seed"] = args.model_seed
        result_print_list["data_seed"] = args.data_seed
        result_print_list["log_lik"] = res.train_history[-1]["log_lik"][0]
        all_print_results.append(result_print_list)

        logging.info("Tree %s", worker.aux)
        logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
        logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    # Save distance data as csv
    with open(args.csv_out, 'w') as csvfile:
        fieldnames = [
                "name",
                "model_seed",
                "data_seed",
                "num_leaves",
                "log_lik",
                UnrootRFDistanceMeasurer.name,
                RootRFDistanceMeasurer.name,
                SPRDistanceMeasurer.name,
                MRCADistanceMeasurer.name,
                "pearson_target_corr",
                "pearson_target_pval",
                "target_lam_dist"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for result_print_list in all_print_results:
            writer.writerow(result_print_list)

    # Save the data
    with open(args.pickle_out, "wb") as f:
        out_dict = {
            "true_model_params": clt_model.get_vars_as_dict(),
            "true_tree": true_tree,
            "res_workers": [(res, worker.aux) for res, worker in successful_res_workers],
            "obs_leaves": obs_leaves,
            "bcode_meta": bcode_meta,
            "args": args}
        if args.use_parsimony:
            out_dict["parsimony_trees"] = parsimony_trees

        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
