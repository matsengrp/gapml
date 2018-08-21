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
        '--init-model-params-file',
        type=str,
        default=None,
        help='pkl file with initializations')
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
        '--dist-to-half-pens',
        type=str,
        default="10.0",
        help="comma separated penalty parameters on the target lambdas")
    parser.add_argument(
        '--tot-time-known',
        action='store_true',
        help='is tot time known?')
    parser.add_argument(
        '--lambda-known',
        action='store_true',
        help='are target rates known?')
    parser.add_argument(
        '--num-tune-splits',
        type=int,
        default=3,
        help="number of random splits of the data for tuning penalty params")
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.5,
        help="fraction of data for training data. for tuning penalty param")
    parser.add_argument('--max-iters', type=int, default=2)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument(
        '--tune-only',
        action='store_true',
        help='only tune penalty params')
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

    assert args.lambda_known or args.tot_time_known
    args.known_params = KnownModelParams(
         target_lams=args.lambda_known,
         tot_time=args.tot_time_known)

    args.dist_to_half_pens = list(sorted(
        [float(lam) for lam in args.dist_to_half_pens.split(",")],
        reverse=True))

    assert args.log_barr >= 0
    assert all(lam >= 0 for lam in args.dist_to_half_pens)
    return args

def fit_tree(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        transition_wrap_maker: TransitionWrapperMaker,
        init_model_param_list: List[Dict],
        oracle_dist_measurers = None,
        known_model_params: KnownModelParams = None,
        num_inits: int = None,
        max_iters: int = None):
    """
    Fits the model params for the given tree and a given penalty param
    """
    worker = LikelihoodScorer(
        get_randint(),
        tree,
        bcode_meta,
        args.max_iters if max_iters is None else max_iters,
        args.num_inits if num_inits is None else num_inits,
        transition_wrap_maker,
        init_model_param_list = init_model_param_list,
        known_params = args.known_params if known_model_params is None else known_model_params,
        dist_measurers = oracle_dist_measurers,
        abundance_weight = args.abundance_weight)
    # TODO: this wastes a lot of time since much of the time is spent here!
    res = worker.run_worker(None)
    return res

def _tune_hyperparams_one_split(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args):
    # First split the data into training vs validation
    train_tree_split, val_tree_split = create_train_val_tree(
            tree,
            bcode_meta,
            args.train_split)

    train_transition_wrap_maker = TransitionWrapperMaker(
            train_tree_split.tree,
            train_tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    val_transition_wrap_maker = TransitionWrapperMaker(
            val_tree_split.tree,
            val_tree_split.bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)

    # Train the model using only the training data
    # First create all the different initialization/optimization settings
    # These will all be fit on the trainin tree
    init_model_param_list = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        if idx == 0:
            new_init_model_params = args.init_params.copy()
        else:
            new_init_model_params = {}
        new_init_model_params["log_barr_pen"] = args.log_barr
        new_init_model_params["dist_to_half_pen"] = dist_to_half_pen
        init_model_param_list.append(new_init_model_params)
    # Actually fit the training tree
    print("fitting train stuff")
    train_results = fit_tree(
        train_tree_split.tree,
        train_tree_split.bcode_meta,
        args,
        train_transition_wrap_maker,
        init_model_param_list)
    logging.info("Finished training all penalty params. Start validation round")

    # Now copy these results over so we can use these fitted values in the validation tree
    init_val_model_param_list = []
    good_idxs = [res is not None for res in train_results]
    for idx, res_train in enumerate(train_results):
        if res_train is not None:
            # Copy over the trained model params except for branch length things
            fixed_params = {}
            for k, v in res_train.model_params_dict.items():
                if bcode_meta.num_barcodes > 1 or (k not in ['branch_len_inners', 'branch_len_offsets_proportion']):
                    fixed_params[k] = v
            # on the validation set, the penalty parameters are all (near) zero
            fixed_params["log_barr_pen"] = 1e-10
            fixed_params["dist_to_half_pen"] = 0
            init_val_model_param_list.append(fixed_params)

    # Now evaluate all these settings on the validation tree
    validation_results = fit_tree(
        val_tree_split.tree,
        val_tree_split.bcode_meta,
        args,
        transition_wrap_maker = val_transition_wrap_maker,
        init_model_param_list = init_val_model_param_list,
        known_model_params = KnownModelParams(
            target_lams = True,
            branch_lens = bcode_meta.num_barcodes > 1,
            indel_params = False, #True,
            tot_time = True),
        # it's a validation tree -- only a single initialization is probably fine
        num_inits = 1,
        max_iters = args.max_iters if bcode_meta.num_barcodes == 1 else 0)

    # Now find the best penalty param by finding the one with the highest log likelihood
    final_validation_results = []
    for idx, dist_to_half_pen in enumerate(args.dist_to_half_pens):
        if good_idxs[idx]:
            # Print results
            res_val = validation_results[int(np.sum(good_idxs[:idx+1])) - 1]
            res_val.dist_to_half_pen = dist_to_half_pen
            # Do not use this as warmstarting values ever.
            if bcode_meta.num_barcodes == 1:
                res_val.model_params_dict.pop('branch_len_inners', None)
                res_val.model_params_dict.pop('branch_len_offsets_proportion', None)
            if res_val is not None:
                final_validation_results.append(res_val)
                logging.info(
                        "Pen param %f val log lik %f",
                        dist_to_half_pen,
                        res_val.log_lik)
                continue
        final_validation_results.append(None)
    return final_validation_results

def tune_hyperparams(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args):
    """
    Tunes the penalty param for the target lambda
    """
    best_params = args.init_params
    best_params["log_barr_pen"] = args.log_barr
    best_params["dist_to_half_pen"] = args.dist_to_half_pens[0]
    if len(args.dist_to_half_pens) == 1:
        return []

    validation_results = []
    for i in range(args.num_tune_splits):
        validation_res = _tune_hyperparams_one_split(
                tree,
                bcode_meta,
                args)
        validation_results.append(validation_res)
    return validation_results

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
    assert (args.true_model_file is None and args.true_collapsed_tree_file is None) or (args.true_model_file is not None and args.true_collapsed_tree_file is not None)
    if args.true_model_file is None or args.true_collapsed_tree_file is None:
        return None, None

    with open(args.true_collapsed_tree_file, "rb") as f:
        collapsed_true_subtree = six.moves.cPickle.load(f)
    with open(args.true_model_file, "rb") as f:
        true_model_dict = six.moves.cPickle.load(f)

    oracle_dist_measurers = TreeDistanceMeasurerAgg([
        UnrootRFDistanceMeasurer,
        RootRFDistanceMeasurer,
        BHVDistanceMeasurer,
        MRCADistanceMeasurer,
        MRCASpearmanMeasurer],
        collapsed_true_subtree,
        args.scratch_dir)
    return true_model_dict, oracle_dist_measurers

def read_init_model_params_file(args, bcode_meta, true_model_dict):
    args.init_params = {
            "target_lams": get_init_target_lams(bcode_meta, 0),
            "boost_softmax_weights": np.ones(3),
            "trim_long_factor": 0.05 * np.ones(2),
            "trim_zero_probs": 0.5 * np.ones(2),
            "trim_short_poissons": 2.5 * np.ones(2),
            "trim_long_poissons": 2.5 * np.ones(2),
            "insert_zero_prob": np.array([0.5]),
            "insert_poisson": np.array([0.5]),
            "double_cut_weight": np.array([0.5]),
            "tot_time": 1,
            "tot_time_extra": 1.3}
    # Use warm-start info if available
    if args.init_model_params_file is not None:
        with open(args.init_model_params_file, "rb") as f:
            args.init_params = six.moves.cPickle.load(f)

    # Copy over true known params if specified
    if args.known_params.tot_time:
        if true_model_dict is not None:
            args.init_params["tot_time"] = true_model_dict["true_model_params"]["tot_time"]
            args.init_params["tot_time_extra"] = true_model_dict["true_model_params"]["tot_time_extra"]
        else:
            args.init_params["tot_time"] = obs_data_dict["time"]
            args.init_params["tot_time_extra"] = 1e-10
    if args.known_params.trim_long_factor:
        args.init_params["trim_long_factor"] = true_model_dict['true_model_params']['trim_long_factor']
    if args.known_params.target_lams:
        args.init_params["target_lams"] = true_model_dict['true_model_params']['target_lams']
        args.init_params["double_cut_weight"] = true_model_dict['true_model_params']['double_cut_weight']

def read_data(args):
    """
    Read the data files...
    """
    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_leaves = obs_data_dict["obs_leaves"]

    obs_leaf = obs_data_dict["obs_leaves"][0]
    no_evts_prop = np.mean([len(evts.events) == 0 for evts in obs_leaf.allele_events_list])
    print("proportion of no events", no_evts_prop)

    evt_set = set()
    for obs in obs_data_dict["obs_leaves"]:
        for evts_list in obs.allele_events_list:
            for evt in evts_list.events:
                evt_set.add(evt)
    logging.info("uniq events %s", evt_set)
    logging.info("num uniq events %d", len(evt_set))
    logging.info("propoertion of double cuts %f", np.mean([e.min_target != e.max_target for e in evt_set]))

    logging.info("Number of uniq obs alleles %d", len(obs_leaves))
    logging.info("Barcode cut sites %s", str(bcode_meta.abs_cut_sites))

    with open(args.topology_file, "rb") as f:
        tree_topology_info = six.moves.cPickle.load(f)
        tree = tree_topology_info["tree"]
        tree.label_node_ids()

    # If this tree is not unresolved, then mark all the multifurcations as resolved
    if not tree_topology_info["multifurc"]:
        for node in tree.traverse():
            node.resolved_multifurcation = True

    logging.info("Tree topology info: %s", tree_topology_info)
    logging.info("Tree topology num leaves: %d", len(tree))

    return bcode_meta, tree, obs_data_dict

def fit_multifurc_tree(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
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
            transition_wraps_multifurc,
            [args.init_params],
            oracle_dist_measurers = oracle_dist_measurers)
    return raw_res[0]

def do_refit_bifurc_tree(
        raw_res: LikelihoodScorerResult,
        bcode_meta: BarcodeMetadata,
        args,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    we need to refit using the bifurcating tree
    """
    # Copy over the latest model parameters for warm start
    param_dict = {k: v for k,v in raw_res.model_params_dict.items()}
    param_dict["log_barr_pen"] = raw_res.log_barr_pen
    param_dict["dist_to_half_pen"] = raw_res.dist_to_half_pen
    refit_bifurc_tree = raw_res.fitted_bifurc_tree.copy()
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
            refit_transition_wrap_maker,
            [param_dict],
            oracle_dist_measurers = oracle_dist_measurers)
    return refit_res[0]

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

    np.random.seed(seed=args.seed)

    if os.path.exists(args.pickle_out):
        logging.info("model exists...")
        return

    # Read input files
    bcode_meta, tree, obs_data_dict = read_data(args)
    true_model_dict, oracle_dist_measurers = read_true_model_files(args)
    read_init_model_params_file(args, bcode_meta, true_model_dict)

    logging.info(tree.get_ascii(attributes=["node_id"], show_internal=True))
    logging.info(tree.get_ascii(attributes=["abundance"], show_internal=True))
    logging.info(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    has_unresolved_multifurcs = check_has_unresolved_multifurcs(tree)
    logging.info("Tree has unresolved mulfirucs? %d", has_unresolved_multifurcs)

    raw_res = None
    refit_res = None

    tune_results = tune_hyperparams(tree, bcode_meta, args)

    if not args.tune_only:
        # Now we can actually train the multifurc tree with the target lambda penalty param fixed
        raw_res = fit_multifurc_tree(
                tree,
                bcode_meta,
                args,
                oracle_dist_measurers)

        # Refit the bifurcating tree if needed
        if not has_unresolved_multifurcs:
            # The tree is already fully resolved. No refitting to do
            refit_res = raw_res
        elif has_unresolved_multifurcs and args.do_refit:
            logging.info("Doing refit")
            refit_res = do_refit_bifurc_tree(
                    raw_res,
                    bcode_meta,
                    args,
                    oracle_dist_measurers)

        #### Mostly a section for printing
        res = refit_res if refit_res is not None else raw_res
        print(res.model_params_dict)
        logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["node_id"], show_internal=True))
        logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["dist"], show_internal=True))
        logging.info(res.fitted_bifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        write_output_json_summary(res, args, oracle_dist_measurers, true_model_dict)

    # Save the data
    with open(args.pickle_out, "wb") as f:
        save_dict = {
                "tune_results": tune_results,
                "raw": raw_res,
                "refit": refit_res}
        six.moves.cPickle.dump(save_dict, f, protocol = 2)

    logging.info("Complete!")

if __name__ == "__main__":
    main()
