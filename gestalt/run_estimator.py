"""
This code fits model parameters and branch lengths
constrained to the multifurcating tree topology.

Searches over bifurcating trees with the same multifurcating tree
by using a continuous parameterization.
Suppose constant events when resolving multifurcations.

Performs training/validation split to pick the target lambda penalty parameter
"""
from __future__ import division, print_function
import os
import sys
import json
import numpy as np
from numpy import ndarray
import argparse
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from scipy.stats import pearsonr
import logging
import six

from optim_settings import KnownModelParams
from transition_wrapper_maker import TransitionWrapperMaker
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from plot_mrca_matrices import plot_mrca_matrix
from split_data import create_train_val_tree
from barcode_metadata import BarcodeMetadata
from tree_distance import *
from constants import *
from common import *

def parse_args():
    parser = argparse.ArgumentParser(description='fit topology and branch lengths for GESTALT')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--topology-file',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help='pkl file with tree topology')
    parser.add_argument(
        '--pickle-out',
        type=str,
        default=None,
        help='pkl file with output')
    parser.add_argument(
        '--true-model-file',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--true-collapsed-tree-file',
        type=str,
        default=None,
        help='pkl file with collapsed tree if available')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--abundance-weight',
        type=float,
        default=0,
        help="weight for abundance")
    parser.add_argument(
        '--log-barr',
        type=float,
        default=0.001,
        help="log barrier parameter on the branch lengths")
    parser.add_argument(
        '--target-lam-pens',
        type=str,
        default="10.0",
        help="comma separated penalty parameters on the target lambdas")
    parser.add_argument(
        '--intercept-lambda-known',
        action='store_true',
        help='are intercept target rates known?')
    parser.add_argument(
        '--tot-time-known',
        action='store_true',
        help='is tot time known?')
    parser.add_argument(
        '--lambda-known',
        action='store_true',
        help='are target rates known?')
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.5,
        help="fraction of data for training data. for tuning penalty param")
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
        '--do-refit',
        action='store_true',
        help='refit after tuning over the compatible bifurcating trees')
    parser.add_argument(
        '--max-sum-states',
        type=int,
        default=None,
        help='maximum number of internal states to marginalize over')
    parser.add_argument(
        '--max-extra-steps',
        type=int,
        default=1,
        help='maximum number of extra steps to explore possible ancestral states')
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default='_output/scratch',
        help='not used at the moment... eventually used by SPR')

    parser.set_defaults(is_refit=False)
    args = parser.parse_args()
    if args.pickle_out is None:
        args.pickle_out = args.topology_file.replace(".pkl", "_fitted.pkl")
        args.json_out = args.topology_file.replace(".pkl", "_fitted.json")
        args.out_folder = os.path.dirname(args.topology_file)
        args.log_file = args.topology_file.replace(".pkl", "_fit_log.txt")
    else:
        args.json_out = args.pickle_out.replace(".pkl", ".json")
        args.out_folder = os.path.dirname(args.pickle_out)
        args.log_file = args.pickle_out.replace(".pkl", "_log.txt")
    print("Log file", args.log_file)
    create_directory(args.pickle_out)

    assert (args.intercept_lambda_known or args.lambda_known) or args.tot_time_known
    args.known_params = KnownModelParams(
         target_lams=args.lambda_known,
         target_lams_intercept=args.intercept_lambda_known,
         tot_time=args.tot_time_known)

    args.target_lam_pens = list(sorted(
        [float(lam) for lam in args.target_lam_pens.split(",")],
        reverse=True))

    assert args.log_barr >= 0
    assert all(lam > 0 for lam in args.target_lam_pens)
    return args

def fit_tree(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        target_lam: float,
        transition_wrap_maker: TransitionWrapperMaker,
        init_model_params: Dict,
        oracle_dist_measurers = None):
    """
    Fits the model params for the given tree and a given penalty param
    """
    worker = LikelihoodScorer(
        get_randint(),
        tree,
        bcode_meta,
        args.log_barr,
        target_lam,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        init_model_params = init_model_params,
        known_params = args.known_params,
        dist_measurers = oracle_dist_measurers,
        abundance_weight = args.abundance_weight)
    res = worker.run_worker(None)
    return res

def tune_hyperparams(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args):
    """
    Tunes the penalty param for the target lambda
    """
    assert not args.known_params.target_lams
    best_params = args.init_params
    if len(args.target_lam_pens) == 1:
        # Nothing to tune.
        best_targ_lam_pen = args.target_lam_pens[0]
        return best_targ_lam_pen, best_params

    train_tree, val_tree, train_bcode_meta, val_bcode_meta = create_train_val_tree(
            tree,
            bcode_meta,
            args.train_split)

    train_transition_wrap_maker = TransitionWrapperMaker(
            train_tree,
            train_bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    val_transition_wrap_maker = TransitionWrapperMaker(
            val_tree,
            val_bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)

    best_targ_lam_pen = args.target_lam_pens[0]
    best_val_log_lik = None
    for i, target_lam_pen in enumerate(args.target_lam_pens):
        logging.info("TUNING %f", target_lam_pen)
        # Train the model using only the training data
        res_train = fit_tree(
            train_tree,
            train_bcode_meta,
            args,
            target_lam_pen,
            train_transition_wrap_maker,
            args.init_model_params)
        logging.info("Done training pen param %f", target_lam_pen)

        # Copy over the trained model params except for branch length things
        fixed_params = {}
        for k, v in res_train.model_params_dict.items():
            if k not in ['branch_len_inners', 'branch_len_offsets_proportion']:
                fixed_params[k] = v

        # Now fit the validation tree with model params fixed
        # TODO: this only keeps the target lambda values fixed and nothing else
        #       We should probably keep more things fixed.
        res_val = fit_tree(
            val_tree,
            val_bcode_meta,
            args,
            0, # target pen lam = zero
            val_transition_wrap_maker,
            init_model_params = fixed_params)
        curr_val_log_lik = res_val.train_history[-1]["log_lik"]
        curr_val_pen_log_lik = res_val.train_history[-1]["pen_log_lik"]
        logging.info(
                "Pen param %f val log lik %f target lams %s",
                target_lam_pen,
                curr_val_log_lik,
                fixed_params['target_lams'])

        if not np.isnan(curr_val_pen_log_lik) and (best_val_log_lik is None or best_val_log_lik < curr_val_log_lik):
            best_val_log_lik = curr_val_log_lik
            best_targ_lam_pen = target_lam_pen
            best_params = fixed_params

    logging.info("Best penalty param %s", best_targ_lam_pen)
    return best_targ_lam_pen, best_params

def get_init_target_lams(bcode_meta, mean_val):
    random_perturb = np.random.uniform(size=bcode_meta.n_targets) * 0.001
    random_perturb = random_perturb - np.mean(random_perturb)
    random_perturb[0] = 0
    return np.exp(mean_val * np.ones(bcode_meta.n_targets) + random_perturb)

def check_has_unresolved_multifurcs(tree: CellLineageTree):
    """
    @return whether or not this tree is an unresolved multifurcating tree
    """
    for node in tree.traverse():
        if not node.is_resolved_multifurcation():
            return True
    return False

def read_true_model_files(args):
    """
    If true model files available, read them
    """
    if args.true_model_file is None or args.true_collapsed_tree_file is None:
        return None, None

    with open(args.true_collapsed_tree_file, "rb") as f:
        collapsed_true_subtree = six.moves.cPickle.load(f)
    with open(args.true_model_file, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)

    oracle_dist_measurers = TreeDistanceMeasurerAgg([
        UnrootRFDistanceMeasurer,
        RootRFDistanceMeasurer,
        #SPRDistanceMeasurer,
        MRCADistanceMeasurer,
        MRCASpearmanMeasurer],
        collapsed_true_subtree,
        args.scratch_dir)
    return true_model_dict, oracle_dist_measurers

def read_data(args):
    """
    Read the data files...
    """
    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]
    logging.info("Number of uniq obs alleles %d", len(obs_leaves))
    logging.info("Barcode cut sites %s", str(bcode_meta.abs_cut_sites))

    with open(args.topology_file, "rb") as f:
        tree_topology_info = six.moves.cPickle.load(f)
        if args.is_refit:
            tree = tree_topology_info.fitted_bifurc_tree
        else:
            tree = tree_topology_info["tree"]
    tree.label_node_ids()

    # If this tree is not unresolved, then mark all the multifurcations as resolved
    if not tree_topology_info["multifurc"]:
        for node in tree.traverse():
            node.resolved_multifurcation = True

    logging.info("Tree topology info: %s", tree_topology_info)
    logging.info("Tree topology num leaves: %d", len(tree))

    return bcode_meta, tree

def fit_multifurc_tree(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        best_targ_lam_pen: float,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    @return LikelihoodScorerResult from fitting model on multifurcating tree
    """
    transition_wraps_multifurc = TransitionWrapperMaker(
            tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    raw_res = fit_tree(
            tree,
            bcode_meta,
            args,
            best_targ_lam_pen,
            transition_wraps_multifurc,
            args.init_params,
            oracle_dist_measurers = oracle_dist_measurers)
    return raw_res

def do_refit_bifurc_tree(
        raw_res: LikelihoodScorerResult,
        bcode_meta: BarcodeMetadata,
        args,
        best_targ_lam_pen: float,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    we need to refit using the bifurcating tree
    """
    # Copy over the latest model parameters for warm start
    print(raw_res.model_params_dict["target_lams"])
    print(raw_res.model_params_dict["target_lams_intercept"])
    param_dict = {k: v for k,v in raw_res.model_params_dict.items()}
    refit_bifurc_tree = raw_res.fitted_bifurc_tree.copy()
    refit_bifurc_tree.label_node_ids()
    print('bifurc')
    print(refit_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
    print(refit_bifurc_tree.get_ascii(attributes=["node_id"], show_internal=True))
    num_nodes = refit_bifurc_tree.get_num_nodes()
    # Copy over the branch lengths
    br_lens = np.zeros(num_nodes)
    for node in refit_bifurc_tree.traverse():
        br_lens[node.node_id] = node.dist
        node.resolved_multifurcation = True
    param_dict["branch_len_inners"] = br_lens
    param_dict["branch_len_offsets_proportion"] = np.zeros(num_nodes)

    # Fit the bifurcating tree
    refit_transition_wrap_maker = TransitionWrapperMaker(
            refit_bifurc_tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    refit_res = fit_tree(
            refit_bifurc_tree,
            bcode_meta,
            args,
            best_targ_lam_pen,
            refit_transition_wrap_maker,
            init_model_params = param_dict,
            oracle_dist_measurers = oracle_dist_measurers)
    return refit_res

def write_output_json_summary(
        res: LikelihoodScorerResult,
        args,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None,
        true_model_dict: Dict = None):
    """
    Writes a json file that summaries what results we got
    """
    if oracle_dist_measurers is not None:
        # do some comparisons against the true model if available
        result_print_dict = oracle_dist_measurers.get_tree_dists([res.fitted_bifurc_tree])[0]
        true_target_lambdas = true_model_dict["true_model_params"]["target_lams"]
        pearson_target = pearsonr(
                true_target_lambdas,
                res.model_params_dict["target_lams"])
        result_print_dict["pearson_target_corr"] = pearson_target[0]
        result_print_dict["pearson_target_pval"] = pearson_target[1]
        result_print_dict["target_lam_dist"] = np.linalg.norm(
                true_target_lambdas - res.model_params_dict["target_lams"])
    else:
        result_print_dict = {}
    result_print_dict["log_lik"] = res.train_history[-1]["log_lik"][0]

    # Save quick summary data as json
    with open(args.json_out, 'w') as outfile:
        json.dump(result_print_dict, outfile)

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    if os.path.exists(args.pickle_out):
        logging.info("model exists...")
        return

    np.random.seed(seed=args.seed)

    # Read input files
    bcode_meta, tree = read_data(args)
    true_model_dict, oracle_dist_measurers = read_true_model_files(args)

    has_unresolved_multifurcs = check_has_unresolved_multifurcs(tree)
    logging.info("Tree has unresolved mulfirucs? %d", has_unresolved_multifurcs)
    logging.info(tree.get_ascii(attributes=["node_id"], show_internal=True))
    logging.info(tree.get_ascii(attributes=["abundance"], show_internal=True))
    logging.info(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    args.init_params = {
            "tot_time": 1,
            "tot_time_extra": 0.3,
            "target_lams_intercept": 0,
            "target_lams": get_init_target_lams(bcode_meta, 0),
            "double_cut_weight": np.array([0.1])}
    if args.known_params.tot_time:
        args.init_params["tot_time"] = true_model_dict["true_model_params"]["tot_time"]
        args.init_params["tot_time_extra"] = true_model_dict["true_model_params"]["tot_time_extra"]
    if args.known_params.target_lams_intercept:
        args.init_params["target_lams_intercept"] = true_model_dict["true_model_params"]["target_lams_intercept"]
        args.init_params["target_lams"] = get_init_target_lams(
                bcode_meta,
                args.init_params["target_lams_intercept"])
        print(true_model_dict["true_model_params"])
    if args.known_params.target_lams:
        args.init_params["target_lams"] = true_model_dict['true_model_params']['target_lams']
        args.init_params["double_cut_weight"] = true_model_dict['true_model_params']['double_cut_weight']
        best_targ_lam_pen = 0
    else:
        # Tune penalty params for the target lambdas
        # Initialize with random values
        best_targ_lam_pen, args.init_params = tune_hyperparams(tree, bcode_meta, args)

    # Now we can actually train the multifurc tree with the target lambda penalty param fixed
    raw_res = fit_multifurc_tree(
            tree,
            bcode_meta,
            args,
            best_targ_lam_pen,
            oracle_dist_measurers)

    # Refit the bifurcating tree if needed
    refit_res = None
    if not has_unresolved_multifurcs:
        # The tree is already fully resolved. No refitting to do
        refit_res = raw_res
    elif has_unresolved_multifurcs and args.do_refit:
        logging.info("Doing refit")
        refit_res = do_refit_bifurc_tree(
                raw_res,
                bcode_meta,
                args,
                best_targ_lam_pen,
                oracle_dist_measurers)

    #### Mostly a section for printing
    res = refit_res if refit_res is not None else raw_res
    print(res.model_params_dict)
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["node_id"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["cell_state"]))

    write_output_json_summary(res, args, oracle_dist_measurers, true_model_dict)

    # Save the data
    with open(args.pickle_out, "wb") as f:
        save_dict = {
                "raw": raw_res,
                "refit": refit_res}
        six.moves.cPickle.dump(save_dict, f, protocol = 2)

    logging.info("Complete!")
    #plot_mrca_matrix(
    #    res.fitted_bifurc_tree,
    #    args.pickle_out.replace(".pkl", "_mrca.png"))

if __name__ == "__main__":
    main()
