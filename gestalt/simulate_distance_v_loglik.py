"""
A simulation engine to see how log likelihood correlates with tree distance
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr, spearmanr
import logging
import six
import random
import seaborn as sns

from clt_estimator import CLTEstimator
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
from tree_manipulation import search_nearby_trees
import ancestral_events_finder as anc_evt_finder
from likelihood_scorer import LikelihoodScorer
from parallel_worker import BatchSubmissionManager
from tree_distance import *

from constants import *
from common import *
from summary_util import *
from simulate_common import *

def parse_args():
    parser = argparse.ArgumentParser(description='simulate GESTALT to see log likelihood vs tree distance')
    parser.add_argument(
        '--out-folder',
        type=str,
        default="_output",
        help='folder to put output in')
    parser.add_argument(
        '--num-moves',
        type=int,
        default=2,
        help="number of steps to perturb away from the true tree")
    parser.add_argument(
        '--num-searches',
        type=int,
        default=1,
        help="number of times to restart the tree perturbations")
    parser.add_argument(
        '--max-explore-trees',
        type=int,
        default=2,
        help="number of trees to consider per distance")
    parser.add_argument(
        '--do-distributed',
        action='store_true',
        help="submit slurm jobs")
    parser.add_argument(
        '--num-barcodes',
        type=int,
        default=1,
        help="number of independent barcodes. we assume all the same")
    parser.add_argument(
        '--target-lambdas',
        type=float,
        nargs=2,
        default=[0.01] * 10,
        help='target cut rates -- will get slightly perturbed for the true value')
    parser.add_argument(
        '--variance-target-lambdas',
        type=float,
        default=0.0005,
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
        '--birth-lambda', type=float, default=2, help='birth rate')
    parser.add_argument(
        '--death-lambda', type=float, default=0.001, help='death rate')
    parser.add_argument(
        '--time', type=float, default=1.6, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.9,
        help='proportion cells sampled/alleles successfully sequenced')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
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
            '--log-barr',
            type=float,
            default=0.2,
            help="log barrier parameter on the branch lengths")
    parser.add_argument(
            '--lasso-param',
            type=float,
            default=0,
            help="lasso parameter on the branch lengths")
    parser.add_argument('--max-iters', type=int, default=1)
    parser.add_argument('--min-leaves', type=int, default=2)
    parser.add_argument('--max-leaves', type=int, default=10)
    parser.add_argument('--max-clt-nodes', type=int, default=10000)

    parser.set_defaults(single_layer=False, two_layers=False, use_cell_state=False)
    args = parser.parse_args()

    args.num_targets = len(args.target_lambdas)
    args.log_file = "%s/distance_v_loglik.log" % args.out_folder
    args.fitted_models_file = "%s/distance_v_loglik_results.pkl" % args.out_folder
    args.scratch_dir = os.path.join(args.out_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)
    print("Log file", args.log_file)
    print("scratch dir", args.scratch_dir)

    return args

def set_branch_lens(tree: CellLineageTree, br_len_vec: ndarray):
    """
    Set the branch lengths in this tree according to br_len_vec
    @param br_len_vec: each element corresponds to a node in the tree, per the node in preorder
    """
    tree.label_node_ids()
    for n in tree.traverse():
        if n.is_root():
            n.dist = 0
        else:
            n.dist = br_len_vec[n.node_id]

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(seed=args.model_seed)

    barcode_orig = BarcodeMetadata.create_fake_barcode_str(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
    bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig, num_barcodes = args.num_barcodes)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    # Set the variance of the perturbations appropriately
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
    logging.info("args.target_lambdas %s (mean %f, var %f)", args.target_lambdas, np.mean(args.target_lambdas), np.var(args.target_lambdas))

    # Create a cell-type tree
    cell_type_tree = create_cell_type_tree(args)

    logging.info(str(args))

    sess = tf.Session()
    with sess.as_default():
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

        # Simulate data
        clt, obs_leaves, true_tree = create_cell_lineage_tree(args, clt_model)
        # Process tree by labeling nodes in the tree
        anc_evt_finder.annotate_ancestral_states(true_tree, bcode_meta)
        true_tree.label_node_ids(CLTLikelihoodModel.NODE_ORDER)
        logging.info("TRUE TREE")
        logging.info(true_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        # Perform NNI moves to find nearby trees
        nearby_trees = [true_tree.copy("deepcopy")]
        for i in range(args.num_searches):
            logging.info("Searching nearby trees, search %d", i)
            nearby_trees += search_nearby_trees(true_tree, max_search_dist=args.num_moves)
        assert len(nearby_trees) > 0

        # Now group trees by distance
        dist_key_measurer = UnrootRFDistanceMeasurer(true_tree, args.scratch_dir)
        # Group nearby trees by the distance measure
        nearby_tree_dict = dist_key_measurer.group_trees_by_dist(nearby_trees, args.max_explore_trees)

        # Random debug check
        oracle_dist = dist_key_measurer.get_dist(true_tree)
        assert oracle_dist == 0

        # Instantiate approximator used by our penalized MLE
        approximator = ApproximatorLB(extra_steps = 1, anc_generations = 1, bcode_metadata = bcode_meta)

        # Make workers for fitting models for each tree
        worker_list = []
        seed = 0
        for rf_dist, uniq_trees in nearby_tree_dict.items():
            logging.info("There are %d trees with RF %d", len(uniq_trees), rf_dist)
            for tree in uniq_trees:
                seed += 1
                lik_scorer = LikelihoodScorer(
                        seed,
                        tree,
                        bcode_meta,
                        cell_type_tree if args.use_cell_state else None,
                        args.know_cell_lambdas,
                        np.array(args.target_lambdas) if args.know_target_lambdas else None,
                        args.log_barr,
                        args.max_iters,
                        approximator,
                        args.time)
                worker_list.append(lik_scorer)

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

        # Process worker results
        final_trees = [worker.tree for _, worker in successful_res_workers]
        pen_log_liks = [res[0][0] for res, _ in successful_res_workers]
        for res, worker in successful_res_workers:
            # Set the branch lengths in the tree
            br_len_vec = res[2]
            set_branch_lens(worker.tree, br_len_vec)
            logging.info("train hist %s", res[-1][-50:])

        # Do final tree distance measurements
        tree_assessor = TreeDistanceMeasurerAgg([
                UnrootRFDistanceMeasurer,
                RootRFDistanceMeasurer,
                SPRDistanceMeasurer,
                MRCADistanceMeasurer],
                true_tree,
                args.scratch_dir)
        final_dist_dicts = tree_assessor.assess(final_trees)

        # Correlation between dist and likelihood among nearby max parsimony trees
        pen_log_liks = np.array(pen_log_liks)
        for dist_name in [measurer.name for measurer in tree_assessor.measurers]:
            final_dists = np.array([d[dist_name] for d in final_dist_dicts])
            final_dists = final_dists[~np.isnan(final_dists)]
            final_pen_log_liks = pen_log_liks[~np.isnan(final_dists)]
            if final_dists.size == 0:
                logging.info("%s:failed", dist_name)
                continue
            logging.info("%s:dists %s", dist_name, str(final_dists))
            logging.info("%s:pen log liks %s", dist_name, str(final_pen_log_liks))
            logging.info("%s:pearson %s", dist_name, pearsonr(final_dists, final_pen_log_liks))
            logging.info("%s:spearman %s", dist_name, spearmanr(final_dists, final_pen_log_liks))
            plt.clf()
            sns.regplot(final_dists, final_pen_log_liks)
            plt.xlabel(dist_name)
            plt.ylabel("pen log lik")
            plt.savefig("%s/dist_vs_ll_%s.png" % (args.out_folder, dist_name))

        with open(args.fitted_models_file, "wb") as f:
            six.moves.cPickle.dump({
                "true_tree": true_tree,
                "true_model": clt_model.get_vars_as_dict(),
                "results": successful_res_workers,
                "dist_dicts": final_dist_dicts,
                "pen_ll": pen_log_liks},
                f,
                protocol=2)

if __name__ == "__main__":
    main()
