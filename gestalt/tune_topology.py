"""
Tunes penalty params, tree topology, and model params
"""
import sys
import six
import os
import argparse
import logging
import numpy as np
import random
from typing import Dict

from cell_lineage_tree import CellLineageTree
from optim_settings import KnownModelParams
from tree_distance import TreeDistanceMeasurerAgg, BHVDistanceMeasurer
from transition_wrapper_maker import TransitionWrapperMaker
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from barcode_metadata import BarcodeMetadata
import hyperparam_tuner
import hanging_chad_finder
from common import create_directory, get_randint


def parse_args():
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--topology-file',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help="Topology file")
    parser.add_argument(
        '--init-model-params-file',
        type=str,
        default=None,
        help='pkl file with initializations')
    parser.add_argument(
        '--out-model-file',
        type=str,
        default="_output/tune_topology_fitted.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log_tune_topology.txt")
    parser.add_argument(
        '--true-model-file',
        type=str,
        default=None,
        help='pkl file with true model if available')
    parser.add_argument(
        '--log-barr',
        type=float,
        default=0.001,
        help="log barrier parameter on the branch lengths")
    parser.add_argument(
        '--dist-to-half-pens',
        type=str,
        default='1',
        help="""
        Comma-separated string with penalty parameters on the target lambdas.
        We will tune over the different penalty params given
        """)
    parser.add_argument(
        '--num-penalty-tune-iters',
        type=int,
        default=1,
        help="""
        Number of iterations to tune the penalty params
        """)
    parser.add_argument(
        '--num-penalty-tune-splits',
        type=int,
        default=2,
        help="""
        Number of random splits of the data for tuning penalty params.
        """)
    parser.add_argument(
        '--num-chad-tune-iters',
        type=int,
        default=2,
        help="""
        Number of iterations to tune the tree topology (aka hanging chads)
        """)
    parser.add_argument(
        '--max-chad-tune-search',
        type=int,
        default=2,
        help="""
        Maximum number of new hanging chad locations to consider at a time
        """)
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--num-inits', type=int, default=1)
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
        '--lambda-known',
        action='store_true',
        help='are target rates known?')
    parser.add_argument(
        '--tot-time-known',
        action='store_true',
        help='is total time known?')
    parser.add_argument(
        '--do-refit',
        action='store_true',
        help='refit the bifurc tree?')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='number of subprocesses to invoke for running')

    parser.set_defaults(tot_time_known=True)
    args = parser.parse_args()

    assert args.log_barr >= 0
    args.dist_to_half_pens = list(sorted(
        [float(lam) for lam in args.dist_to_half_pens.split(",")],
        reverse=True))

    create_directory(args.out_model_file)
    args.topology_folder = os.path.dirname(args.topology_file)
    args.scratch_dir = os.path.join(
            args.topology_folder,
            'scratch')
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.known_params = KnownModelParams(
         target_lams=args.lambda_known,
         tot_time=args.tot_time_known)

    assert args.num_penalty_tune_iters >= 1
    assert args.num_chad_tune_iters >= args.num_penalty_tune_iters
    return args


def get_init_target_lams(bcode_meta, mean_val):
    random_perturb = np.random.uniform(size=bcode_meta.n_targets) * 0.001
    random_perturb = random_perturb - np.mean(random_perturb)
    random_perturb[0] = 0
    return np.exp(mean_val * np.ones(bcode_meta.n_targets) + random_perturb)


def read_init_model_params_file(args, bcode_meta, obs_data_dict, true_model_dict):
    args.init_params = {
            "target_lams": get_init_target_lams(bcode_meta, 0),
            "boost_softmax_weights": np.array([1, 2, 2]),
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

    args.init_params["log_barr_pen"] = args.log_barr
    args.init_params["dist_to_half_pen"] = args.dist_to_half_pens[0]

    # Copy over true known params if specified
    if args.known_params.tot_time:
        if true_model_dict is not None:
            args.init_params["tot_time"] = true_model_dict["tot_time"]
            args.init_params["tot_time_extra"] = true_model_dict["tot_time_extra"]
        else:
            args.init_params["tot_time"] = obs_data_dict["time"]
            args.init_params["tot_time_extra"] = 1e-10
    if args.known_params.trim_long_factor:
        args.init_params["trim_long_factor"] = true_model_dict['trim_long_factor']
    if args.known_params.target_lams:
        args.init_params["target_lams"] = true_model_dict['target_lams']
        args.init_params["double_cut_weight"] = true_model_dict['double_cut_weight']


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


def read_true_model_files(args, num_barcodes):
    """
    If true model files available, read them
    """
    if args.true_model_file is None:
        return None, None

    # TODO: take the tree loading comparison code out of plot code
    true_model_dict, true_tree, _ = plot_simulation_common.get_true_model(
            args.true_model_file,
            None,
            num_barcodes)

    oracle_dist_measurers = TreeDistanceMeasurerAgg(
        [BHVDistanceMeasurer],
        true_tree,
        args.scratch_dir)
    return true_model_dict, oracle_dist_measurers


def has_unresolved_multifurcs(tree: CellLineageTree):
    """
    @return whether or not this tree is an unresolved multifurcating tree
    """
    for node in tree.traverse():
        if not node.is_resolved_multifurcation():
            return True
    return False


def fit_multifurc_tree(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        args,
        param_dict: Dict,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    @return LikelihoodScorerResult from fitting model on multifurcating tree
    """
    transition_wrap_maker = TransitionWrapperMaker(
            tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    result = LikelihoodScorer(
        get_randint(),
        tree,
        bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        init_model_param_list=[param_dict],
        known_params=args.known_params,
        dist_measurers=oracle_dist_measurers).run_worker(None)[0]
    return result

def do_refit_bifurc_tree(
        raw_res: LikelihoodScorerResult,
        bcode_meta: BarcodeMetadata,
        args,
        oracle_dist_measurers: TreeDistanceMeasurerAgg = None):
    """
    we need to refit using the bifurcating tree
    """
    # Copy over the latest model parameters for warm start
    param_dict = raw_res.get_fit_params()
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
    transition_wrap_maker = TransitionWrapperMaker(
            raw_res.fitted_bifurc_tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    bifurc_res = LikelihoodScorer(
        get_randint(),
        raw_res.fitted_bifurc_tree,
        bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        init_model_param_list=[param_dict],
        known_params=args.known_params,
        dist_measurers=oracle_dist_measurers).run_worker(None)[0]
    return bifurc_res


def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    bcode_meta, tree, obs_data_dict = read_data(args)
    true_model_dict, oracle_dist_measurers = read_true_model_files(args, bcode_meta.num_barcodes)
    read_init_model_params_file(args, bcode_meta, obs_data_dict, true_model_dict)

    print(tree.get_ascii(attributes=["allele_events_list_str"]))
    print(tree.get_ascii(attributes=["node_id"]))

    # Find hanging chads
    hanging_chads = hanging_chad_finder.get_chads(tree)
    has_chads = len(hanging_chads) > 0
    logging.info("total number of hanging chads %d", len(hanging_chads))
    logging.info(hanging_chads)

    # Begin tuning
    tuning_history = []
    for i in range(args.num_chad_tune_iters):
        penalty_tune_result = None
        if i < args.num_penalty_tune_iters:
            if len(args.dist_to_half_pens) == 1:
                # If nothing to tune... do nothing
                init_model_params = args.init_params
            else:
                # Tune penalty params!
                logging.info("Iter %d: Tuning penalty params", i)
                penalty_tune_result = hyperparam_tuner.tune(tree, bcode_meta, args)
                init_model_params, best_res = penalty_tune_result.get_best_result()

        # pick a chad at random
        chad_tune_result = None
        if has_chads:
            # Now tune the hanging chads!
            random_chad = random.choice(hanging_chads)
            logging.info("Iter %d: Tuning chad %s", i, random_chad)
            chad_tune_result = hanging_chad_finder.tune(
                random_chad,
                tree,
                bcode_meta,
                args,
                init_model_params,
                oracle_dist_measurers,
            )
            tree, init_model_params, best_res = chad_tune_result.get_best_result()
        else:
            best_res = fit_multifurc_tree(
                    tree,
                    bcode_meta,
                    args,
                    init_model_params,
                    oracle_dist_measurers)

        tuning_history.append({
            "chad_tune_result": chad_tune_result,
            "penalty_tune_result": penalty_tune_result,
        })
        if not has_chads:
            break

    # Tune the final bifurcating tree one more time?
    bifurc_res = None
    if not has_unresolved_multifurcs(tree):
        bifurc_res = best_res
    elif args.do_refit:
        bifurc_res = do_refit_bifurc_tree(
                best_res,
                bcode_meta,
                args,
                oracle_dist_measurers)

    # Save results
    with open(args.out_model_file, "wb") as f:
        result = {
            "tuning_history": tuning_history,
            "bifurc_res": bifurc_res,
        }
        six.moves.cPickle.dump(result, f, protocol=2)
    logging.info("Complete!!!")


if __name__ == "__main__":
    main()
