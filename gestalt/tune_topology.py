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
from model_assessor import ModelAssessor
from tree_distance import BHVDistanceMeasurer, InternalCorrMeasurer
from transition_wrapper_maker import TransitionWrapperMaker
from likelihood_scorer import LikelihoodScorer, LikelihoodScorerResult
from barcode_metadata import BarcodeMetadata
import hyperparam_tuner
import hanging_chad_finder
from common import create_directory, get_randint, save_data, get_init_target_lams, parse_comma_str
import file_readers
import collapsed_tree


def parse_args(args):
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
        '--target-lam-pen-params',
        type=str,
        default='1',
        help="""
        Comma-separated string with penalty parameters on the target lambdas.
        We will tune over the different penalty params given
        """)
    parser.add_argument(
        '--branch-pen-params',
        type=str,
        default='1',
        help="""
        Comma-separated string with penalty parameters on the branch penalty
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
        '--max-fit-splits',
        type=int,
        default=5,
        help="""
        Maximum number of random splits to fit on the data
        """)
    parser.add_argument(
        '--num-penalty-tune-splits',
        type=int,
        default=2,
        help="""
        Number of random splits of the data for tuning penalty params.
        """)
    parser.add_argument(
        '--num-chad-stop',
        type=int,
        default=2,
        help="""
        If we select don't change the tree topology for this many steps in a row,
        then we stop the hanging chad tuner.
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
        '--lambda-decay-known',
        action='store_true',
        help='are target rate decay rates known?')
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
    parser.add_argument(
        '--num-init-random-rearrange',
        type=int,
        default=0,
        help='number of times we randomly rearrange tree at the beginning')
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default=None)
    parser.add_argument(
        '--count-chads',
        action='store_true',
        help="""
        Log the number of hanging chads found in the tree
        """)
    parser.add_argument(
        '--use-poisson',
        action='store_true',
        help="Use poisson distribution")

    parser.set_defaults(tot_time_known=True)
    args = parser.parse_args(args)

    args.branch_pen_params = list(sorted(
        parse_comma_str(args.branch_pen_params),
        reverse=True))
    assert all([p > 0 for p in args.branch_pen_params])
    args.target_lam_pen_params = list(sorted(
        parse_comma_str(args.target_lam_pen_params),
        reverse=True))
    assert all([p > 0 for p in args.target_lam_pen_params])

    create_directory(args.out_model_file)
    if args.scratch_dir is None:
        topology_folder = os.path.dirname(args.topology_file)
        args.scratch_dir = os.path.join(topology_folder, "scratch")
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.known_params = KnownModelParams(
         target_lams=args.lambda_known,
         target_lam_decay_rate=args.lambda_decay_known,
         tot_time=args.tot_time_known)

    assert args.num_penalty_tune_iters >= 1
    assert args.tot_time_known
    assert args.num_chad_tune_iters >= args.num_penalty_tune_iters
    return args


def read_fit_params_file(args, bcode_meta, obs_data_dict, true_model_dict):
    fit_params = {
            "target_lams": get_init_target_lams(bcode_meta.n_targets, 0) * 0.5,
            "target_lam_decay_rate": np.array([0.1]),
            "boost_softmax_weights": np.array([1, 2, 2]),
            "trim_long_factor": 0.05 * np.ones(2),
            "trim_zero_probs": 0.5 * np.ones(4),
            "trim_short_params":  np.array([1,1]) if args.use_poisson else np.array([0.7,0.2] * 2),
            "trim_long_params":  np.array([1,1]) if args.use_poisson else np.array([0.7,0.2] * 2),
            "insert_zero_prob": np.array([0.5]),
            "insert_params": np.array([1]) if args.use_poisson else np.array([1,0.1]),
            "double_cut_weight": np.array([0.1]),
            "tot_time": 1,
            "tot_time_extra": 1.3}
    # Use warm-start info if available
    if args.init_model_params_file is not None:
        with open(args.init_model_params_file, "rb") as f:
            fit_params = six.moves.cPickle.load(f)

    # Copy over true known params if specified
    if args.known_params.tot_time:
        if true_model_dict is not None:
            fit_params["tot_time"] = true_model_dict["tot_time"]
            fit_params["tot_time_extra"] = true_model_dict["tot_time_extra"]
        else:
            fit_params["tot_time"] = obs_data_dict["time"]
            fit_params["tot_time_extra"] = 1e-10
    if args.known_params.trim_long_factor:
        fit_params["trim_long_factor"] = true_model_dict['trim_long_factor']
    if args.known_params.target_lam_decay_rate:
        fit_params["target_lam_decay_rate"] = true_model_dict['target_lam_decay_rate']
    if args.known_params.target_lams:
        fit_params["target_lams"] = true_model_dict['target_lams']
        fit_params["double_cut_weight"] = true_model_dict['double_cut_weight']
    if args.known_params.indel_params:
        fit_params["boost_softmax_weights"] = true_model_dict['boost_softmax_weights']
        fit_params["trim_zero_probs"] = true_model_dict['trim_zero_probs']
        fit_params["insert_zero_prob"] = true_model_dict['insert_zero_prob']
    if args.known_params.indel_dists:
        fit_params["trim_short_params"] = true_model_dict['trim_short_params']
        fit_params["trim_long_params"] = true_model_dict['trim_long_params']
        fit_params["insert_params"] = true_model_dict['insert_params']
    return fit_params


def read_data(args):
    """
    Read the data files...
    """
    tree, obs_data_dict = file_readers.read_data(args.obs_file, args.topology_file)
    bcode_meta = obs_data_dict["bcode_meta"]

    # This section just prints interesting things...
    evt_set = set()
    for obs in obs_data_dict["obs_leaves"]:
        for evts_list in obs.allele_events_list:
            for evt in evts_list.events:
                evt_set.add(evt)
    logging.info("Num uniq events %d", len(evt_set))
    logging.info("Proportion of double cuts %f", np.mean([e.min_target != e.max_target for e in evt_set]))
    abundances = [obs.abundance for obs in obs_data_dict["obs_leaves"]]
    logging.info("Range of abundance vals %d %d (mean %f)", np.min(abundances), np.max(abundances), np.mean(abundances))
    if tree is not None:
        logging.info("Number of leaves %d", len(tree))

    return bcode_meta, tree, obs_data_dict


def read_true_model_files(args, num_barcodes, measurer_classes=None):
    """
    If true model files available, read them
    """
    if args.true_model_file is None:
        return None, None

    if measurer_classes == None:
        measurer_classes = [BHVDistanceMeasurer, InternalCorrMeasurer]

    true_model_dict, assessor = file_readers.read_true_model(
            args.true_model_file,
            num_barcodes,
            measurer_classes=measurer_classes,
            scratch_dir=args.scratch_dir)

    return true_model_dict, assessor


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
        assessor: ModelAssessor = None):
    """
    @return LikelihoodScorerResult from fitting model on multifurcating tree
    """
    transition_wrap_maker = TransitionWrapperMaker(
            tree,
            bcode_meta,
            args.max_extra_steps,
            args.max_sum_states)
    if 'branch_len_inners' in param_dict:
        # If branch length estimates are provided and we have the mapping between
        # the full_tree nodes and the nodes in the no_chad tree, then we should do warm-start.
        full_tree_br_inners = param_dict['branch_len_inners']
        full_tree_br_offsets = param_dict['branch_len_offsets_proportion']
        num_nodes = tree.get_num_nodes()
        tree_br_len_inners = np.zeros(num_nodes)
        tree_br_len_offsets = np.ones(num_nodes) * 0.4 + np.random.rand() * 0.1

        # Mark the nodes to get the corresponding node_id in the full_tree
        for node in tree.traverse():
            node.add_feature("orig_node_id", node.node_id)

        tree.label_node_ids()

        node_mapping = {}
        for node in tree.traverse():
            node_mapping[node.node_id] = node.orig_node_id
            node.del_feature("orig_node_id")

        for node in tree.traverse():
            if node.node_id in node_mapping and node_mapping[node.node_id] in full_tree_br_inners:
                tree_br_len_inners[node.node_id] = full_tree_br_inners[node_mapping[node.node_id]]
                tree_br_len_offsets[node.node_id] = full_tree_br_offsets[node_mapping[node.node_id]]
        param_dict['branch_len_inners'] = tree_br_len_inners
        param_dict['branch_len_offsets_proportion'] = tree_br_len_offsets

    result = LikelihoodScorer(
        get_randint(),
        tree,
        bcode_meta,
        args.max_iters,
        args.num_inits,
        transition_wrap_maker,
        fit_param_list=[param_dict],
        known_params=args.known_params,
        scratch_dir=args.scratch_dir,
        use_poisson=args.use_poisson,
        assessor=assessor).run_worker(None)[0]
    return result

def _do_random_rearrange(tree, bcode_meta, num_random_rearrange):
    """
    @param num_random_rearrange: number of times to take a random chad and randomly regraft
    Picks a random chad and randomly places it under a possible parent
    """
    orig_num_leaves = len(tree)
    recent_chads = set()
    for i in range(num_random_rearrange):
        print("doing random rearrange", i)
        random_chad = hanging_chad_finder.get_random_chad(
                tree,
                bcode_meta,
                exclude_chad_func=lambda node: make_chad_psuedo_id(node) in recent_chads)
        if random_chad is None:
            logging.info("No hanging chad to be found")
            return tree

        rand_chad_id = make_chad_psuedo_id(random_chad.node)
        if rand_chad_id in recent_chads:
            # Reset the recent chad list since the random chad finder failed
            recent_chads = set()
        recent_chads.add(rand_chad_id)

        logging.info(tree.get_ascii(attributes=["anc_state_list_str"]))
        logging.info(str(random_chad))

        # Pick random equal parsimony tree
        new_tree = random.choice(random_chad.possible_full_trees)

        # Remove any unifurcations that may have been introduced when we
        # detached the hanging chad
        collapsed_tree._remove_single_child_unobs_nodes(new_tree)

        new_tree.label_node_ids()
        assert orig_num_leaves == len(new_tree)
        tree = new_tree

    return tree

def make_chad_psuedo_id(node: CellLineageTree):
    return tuple([str(a) for a in node.anc_state_list])

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    # Load data
    bcode_meta, tree, obs_data_dict = read_data(args)
    true_model_dict, assessor = read_true_model_files(args, bcode_meta.num_barcodes)
    fit_params = read_fit_params_file(args, bcode_meta, obs_data_dict, true_model_dict)

    logging.info("num barcodes %d", bcode_meta.num_barcodes)
    logging.info("STARTING.... before random rearrange")
    logging.info(tree.get_ascii(attributes=["allele_events_list_str"]))
    logging.info(tree.get_ascii(attributes=["node_id"]))

    num_all_chads = None
    if args.count_chads:
        all_chad_sketches = hanging_chad_finder.get_all_chads(
            tree,
            bcode_meta,
            max_possible_trees=2)
        logging.info("Total of %d chads found", len(all_chad_sketches))
        num_all_chads = len(all_chad_sketches)
    tree = _do_random_rearrange(tree, bcode_meta, args.num_init_random_rearrange)

    logging.info("STARTING for reals!")
    logging.info(tree.get_ascii(attributes=["allele_events_list_str"]))
    logging.info(tree.get_ascii(attributes=["node_id"]))
    logging.info("Abundance...")
    logging.info(tree.get_ascii(attributes=["abundance"]))
    for leaf in tree:
        assert leaf.abundance >= 1

    for node in tree.traverse():
        assert node.node_id is not None

    # Begin tuning
    tuning_history = []
    recent_chads = set()
    num_stable = 0
    for i in range(args.num_chad_tune_iters):
        np.random.seed(args.seed + i + 1)
        random.seed(args.seed + i + 1)

        penalty_tune_result = None
        best_res = None
        if i < args.num_penalty_tune_iters:
            if len(args.branch_pen_params) == 1 and len(args.target_lam_pen_params) == 1:
                # If nothing to tune... do nothing
                fit_params["branch_pen_param"] = args.branch_pen_params[0]
                fit_params["target_lam_pen_param"] = args.target_lam_pen_params[0]
            else:
                # Tune penalty params!
                logging.info("Iter %d: Tuning penalty params", i)
                penalty_tune_result = hyperparam_tuner.tune(tree, bcode_meta, args, fit_params, assessor)
                _, fit_params, best_res = penalty_tune_result.get_best_result()
            logging.info("Iter %d: Best pen param %f %s", i, fit_params["branch_pen_param"], fit_params["target_lam_pen_param"])

        # Find hanging chads
        # TODO: kind slow right now... reruns chad-finding code
        # cause nodes are getting renumbered...
        random_chad = None
        if args.max_chad_tune_search >= 1:
            logging.info("chad finding time")
            random_chad = hanging_chad_finder.get_random_chad(
                    tree,
                    bcode_meta,
                    exclude_chad_func=lambda node: make_chad_psuedo_id(node) in recent_chads,
                    masking_only=True)
        has_chads = random_chad is not None
        if not has_chads:
            tuning_history.append({
                "chad_tune_result": None,
                "penalty_tune_result": penalty_tune_result,
                "best_res": best_res,
            })
            save_data(tuning_history, args.out_model_file)
            logging.info("No hanging chads found")
            break

        # Mark which chads we've seen recently
        rand_chad_id = make_chad_psuedo_id(random_chad.node)
        # Reset the recent chad list since the random chad finder failed
        if rand_chad_id in recent_chads:
            recent_chads = set()
        recent_chads.add(rand_chad_id)

        # Now tune the hanging chads!
        logging.info(
                "Iter %d: Tuning chad %s",
                i,
                random_chad)
        num_old_leaves = len(tree)
        chad_tune_result, is_same = hanging_chad_finder.tune(
            random_chad,
            args.max_chad_tune_search,
            tree,
            bcode_meta,
            args,
            fit_params,
            assessor,
        )
        tree, fit_params, best_res = chad_tune_result.get_best_result()
        if is_same:
            num_stable += int(is_same)
        else:
            num_stable = 0
        print("num stable", num_stable)

        # just for fun... check that the number of leaves match
        assert len(tree) == num_old_leaves

        if assessor is not None:
            logging.info(
                    "Iter %d, begin dists %s, log lik %f %f",
                    i,
                    best_res.train_history[0]["performance"],
                    best_res.train_history[0]["pen_log_lik"],
                    best_res.train_history[0]["log_lik"])
            logging.info(
                    "Iter %d, end dists %s, log lik %f %f (num iters %d)",
                    i,
                    best_res.train_history[-1]["performance"],
                    best_res.pen_log_lik,
                    best_res.log_lik,
                    len(best_res.train_history) - 1)

        tuning_history.append({
            "chad_tune_result": chad_tune_result,
            "penalty_tune_result": penalty_tune_result,
            "best_res": best_res,
        })
        save_data(tuning_history, args.out_model_file)
        if (num_all_chads is not None and num_stable >= num_all_chads) or num_stable >= args.num_chad_stop:
            logging.info("Hanging chad tuner has converged")
            break

    logging.info("Done tuning chads!")
    last_chad_res = tuning_history[-1]['chad_tune_result']
    if last_chad_res is None or last_chad_res.num_chad_leaves > 1:
        final_fit = fit_multifurc_tree(
                tree,
                bcode_meta,
                args,
                fit_params,
                assessor)
    else:
        final_fit = best_res

    # Save results
    with open(args.out_model_file, "wb") as f:
        result = {
            "tuning_history": tuning_history,
            "final_fit": final_fit,
        }
        six.moves.cPickle.dump(result, f, protocol=2)
    logging.info("Complete!!!")


if __name__ == "__main__":
    main()
