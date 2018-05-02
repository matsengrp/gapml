"""
A simulation engine to see how well cell lineage estimation performs
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
from clt_likelihood_estimator import *
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
        '--know-target-lambdas',
        action='store_true')
    parser.add_argument(
        '--know-cell-lambdas',
        action='store_true')
    parser.add_argument(
        '--const-branch-len',
        action='store_true')
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
    parser.add_argument(
            '--optim-seed',
            type=int,
            default=40,
            help="Seed for generating the model")
    parser.add_argument(
            '--log-barr',
            type=float,
            default=0.001,
            help="log barrier parameter on the branch lengths")
    parser.add_argument(
            '--lasso-param',
            type=float,
            default=0,
            help="lasso parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=10)
    parser.add_argument('--max-clt-nodes', type=int, default=40000)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
            '--mix-path',
            type=str,
            default=MIX_PATH)
    parser.add_argument('--max-trees',
            type=int,
            default=2)
    parser.add_argument('--num-jumbles',
            type=int,
            default=1)
    parser.add_argument('--use-parsimony', action='store_true', help="use mix (CS parsimony) to estimate tree topologies")
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

    if args.use_parsimony:
        # check that there is no infile in the current folder -- this will
        # screw up mix because it will use the wrong input file
        my_file = Path("infile")
        assert not my_file.exists()

    return args

def collapse_internally_labelled_tree(tree: CellLineageTree):
    coll_tree = tree.copy("deepcopy")
    for n in coll_tree.traverse():
        n.name = n.allele_events_list_str
        if not n.is_root():
            if n.allele_events_list_str == n.up.allele_events_list_str:
                n.dist = 0
            else:
                n.dist = 1
    coll_tree = collapse_zero_lens(coll_tree)
    return coll_tree

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

    clt, obs_leaves, true_tree = create_cell_lineage_tree(
            args,
            clt_model,
            bifurcating_only=True)

    tree_dist_measurers = TreeDistanceMeasurerAgg([
            UnrootRFDistanceMeasurer,
            RootRFDistanceMeasurer,
            SPRDistanceMeasurer,
            MRCADistanceMeasurer],
            true_tree,
            args.scratch_dir)

    # Gather true branch lengths
    true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)
    leaf_strs = [l.allele_events_list_str for l in true_tree]
    assert len(leaf_strs) == len(set(leaf_strs))

    for leaf in true_tree:
        assert np.isclose(leaf.get_distance(true_tree), args.time)

    # Get parsimony score of tree?
    pars_score = true_tree.get_parsimony_score()
    logging.info("Oracle tree parsimony score %d", pars_score)

    # Print fun facts about the data
    num_leaves = len(true_tree)
    logging.info("True tree topology, num leaves %d" % num_leaves)
    logging.info(true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(true_tree.get_ascii(attributes=["cell_state"], show_internal=True))
    logging.info(true_tree.get_ascii(attributes=["observed"], show_internal=True))
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))

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

    trees_to_test = {}

    # Get the parsimony-estimated topologies
    if args.use_parsimony:
        measurer = UnrootRFDistanceMeasurer(true_tree, args.scratch_dir)
        parsimony_trees = get_parsimony_trees(
            obs_leaves,
            args,
            bcode_meta,
            do_collapse=False)

        min_dist = measurer.get_dist(parsimony_trees[0])
        best_parsimony_tree = parsimony_trees[0]
        parsimony_dists = [min_dist]
        for pars_tree in parsimony_trees:
            tree_dist = measurer.get_dist(pars_tree)
            parsimony_dists.append(tree_dist)
            if tree_dist < min_dist:
                best_parsimony_tree = pars_tree
                min_dist = tree_dist
        logging.info("Uniq parsimony %s distances: %s", measurer.name, np.unique(parsimony_dists))
        logging.info("Mean parsimony %s distance: %f", measurer.name, np.mean(parsimony_dists))
        logging.info("Min parsimony %s distance: %d", measurer.name, min_dist)

        trees_to_test["best_parsimony"] = best_parsimony_tree
        logging.info("Best parsimony -- not collapsed")
        logging.info(best_parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        best_coll_tree = collapse_internally_labelled_tree(best_parsimony_tree)
        trees_to_test["best_pars_multifurc"] = best_coll_tree
        logging.info("Best parsimony -- collapsed")
        logging.info(best_coll_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        random_parsimony_tree = parsimony_trees[np.random.randint(low=0, high=len(parsimony_trees))]
        trees_to_test["random_parsimony"] = random_parsimony_tree
        logging.info("Random parsimony -- not collapsed")
        logging.info(random_parsimony_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

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

    # Collapse the oracle tree
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
        result_print_list["log_lik"] = res.train_history[-1]["log_lik"]
        all_print_results.append(result_print_list)

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
            "res_workers": successful_res_workers,
            "obs_leaves": obs_leaves,
            "bcode_meta": bcode_meta,
            "args": args}
        if args.use_parsimony:
            out_dict["parsimony_trees"] = parsimony_trees

        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
