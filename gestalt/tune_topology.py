"""
Tunes over different multifurcating topologies.
Basically a wrapper around `run_estimator.py`
This will create jobs to run on the cluster for each of the trees
"""
import sys
import six
import os
import glob
import argparse
import logging
import six
import numpy as np
from typing import List, Tuple, Dict

from parallel_worker import BatchSubmissionManager
from estimator_worker import RunEstimatorWorker
from likelihood_scorer import LikelihoodScorerResult
from common import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description='tune over many multifurcating topologies and fit model parameters')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--topology-file-template',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help="""
        We look in the directory of this path for any trees that need to be processed.
        We replace the 0 with a *[0-9] when we grep for matching trees.
        """)
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
        '--true-collapsed-tree-file',
        type=str,
        default=None,
        help='pkl file with collapsed tree if available')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--log-barr',
        type=float,
        default=0.001,
        help="log barrier parameter on the branch lengths")
    parser.add_argument(
        '--dist-to-half-pens',
        type=str,
        default='1',
        help="comma-separated string with penalty parameters on the target lambdas")
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.75,
        help="fraction of data for training data. for tuning penalty param")
    parser.add_argument(
        '--total-tune-splits',
        type=int,
        default=1,
        help="""
        number of random splits of the data for tuning penalty params
        across all topologies we tune
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
        '--cpu-threads',
        type=int,
        default=6,
        help='number of cpu threads to request in srun when submitting jobs')
    parser.add_argument(
        '--submit-srun',
        action='store_true',
        help='is using slurm to submit jobs')
    parser.add_argument(
        '--lambda-known',
        action='store_true',
        help='are target rates known?')
    parser.add_argument(
        '--tot-time-known',
        action='store_true',
        help='is total time known?')
    parser.add_argument(
        '--max-topologies',
        type=int,
        default=10,
        help='max topologies to tune over')

    parser.set_defaults()
    args = parser.parse_args()

    assert args.log_barr >= 0
    args.dist_to_half_pen_list = list(sorted(
        [float(lam) for lam in args.dist_to_half_pens.split(",")],
        reverse=True))

    create_directory(args.out_model_file)
    args.topology_folder = os.path.dirname(args.topology_file_template)
    args.scratch_dir = os.path.join(
            args.topology_folder,
            'scratch')
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args

def get_best_hyperparam(
        args,
        tune_results: List[Tuple[Dict, RunEstimatorWorker]]):

    total_pen_param_score = np.zeros(len(args.dist_to_half_pen_list))
    for (topology_res, _) in tune_results:
        for topology_rep_res in topology_res["tune_results"]:
            for i, res in enumerate(topology_rep_res):
                total_pen_param_score[i] += res.score

    logging.info("Total tuning scores %s", total_pen_param_score)
    best_idx = np.argmax(total_pen_param_score)
    best_dist_to_half_pen = args.dist_to_half_pen_list[best_idx]
    logging.info("Best penalty param %s", best_dist_to_half_pen)
    # Create the initialization model params for warm starting
    warm_starts = [
            topology_res["tune_results"][0][best_idx].model_params_dict
            for topology_res, _ in tune_results]

    return best_dist_to_half_pen, warm_starts

def tune_hyperparams(
        topology_files,
        args):
    worker_list = []
    # TODO: right now we only ever do one split for training validation
    # ... mostly cause we're too lazy to implement something else
    num_tune_splits = 1 if args.num_barcodes > 1 else args.total_tune_splits

    for file_idx, top_file in enumerate(topology_files):
        worker = RunEstimatorWorker(
            args.obs_file,
            top_file,
            args.out_model_file.replace(".pkl", "_tune_only_tree%d.pkl" % file_idx),
            None,
            args.true_model_file,
            args.true_collapsed_tree_file,
            args.seed + file_idx,
            args.log_barr,
            args.dist_to_half_pens,
            args.max_iters,
            # When tuning hyper-params, we just use one initialization
            num_inits = 1,
            lambda_known = args.lambda_known,
            tot_time_known = args.tot_time_known,
            do_refit = False,
            tune_only = True,
            max_sum_states = args.max_sum_states,
            max_extra_steps = args.max_extra_steps,
            train_split = args.train_split,
            num_tune_splits = num_tune_splits,
            scratch_dir = args.scratch_dir)
        worker_list.append(worker)

    if args.submit_srun:
        logging.info("Submitting jobs")
        job_manager = BatchSubmissionManager(
                worker_list,
                None,
                len(worker_list),
                args.scratch_dir,
                threads=args.cpu_threads)
        successful_workers = job_manager.run(successful_only=True)
        assert len(successful_workers) > 0
    else:
        logging.info("Running locally")
        successful_workers = [(w.run_worker(None), w) for w in worker_list]

    best_hyperparam, warm_starts = get_best_hyperparam(
            args,
            successful_workers)

    # Create warm start model params file
    warm_start_files = []
    for file_idx, (top_file, warm_start) in enumerate(zip(topology_files, warm_starts)):
        warm_start_file_name = args.out_model_file.replace(".pkl", "_warm%d.pkl" % file_idx)
        warm_start_files.append(warm_start_file_name)
        with open(warm_start_file_name, "wb") as f:
            print("WARMSART", warm_start)
            six.moves.cPickle.dump(warm_start, f, protocol=2)

    return best_hyperparam, warm_start_files

def fit_models(
        topology_files,
        warm_start_files,
        args,
        pen_param):
    worker_list = []
    for file_idx, (top_file, warm_start_file) in enumerate(zip(topology_files, warm_start_files)):
        worker = RunEstimatorWorker(
            args.obs_file,
            top_file,
            args.out_model_file.replace(".pkl", "_refitnew_tree%d.pkl" % file_idx),
            warm_start_file,
            args.true_model_file,
            args.true_collapsed_tree_file,
            args.seed + file_idx,
            args.log_barr,
            str(pen_param),
            args.max_iters,
            num_inits = args.num_inits,
            lambda_known = args.lambda_known,
            tot_time_known = args.tot_time_known,
            do_refit = True,
            tune_only = False,
            max_sum_states = args.max_sum_states,
            max_extra_steps = args.max_extra_steps,
            train_split = 1,
            num_tune_splits = 1,
            scratch_dir = args.scratch_dir)
        worker_list.append(worker)

    if args.submit_srun:
        logging.info("Submitting jobs")
        job_manager = BatchSubmissionManager(
                worker_list,
                None,
                len(worker_list),
                args.scratch_dir,
                threads=args.cpu_threads)
        successful_workers = job_manager.run(successful_only=True)
        assert len(successful_workers) > 0
    else:
        logging.info("Running locally")
        successful_workers = [(w.run_worker(None), w) for w in worker_list]

    best_pen_log_lik = successful_workers[0][0]["refit"].pen_log_lik
    best_worker = successful_workers[0][1]
    for (result, succ_worker) in successful_workers[1:]:
        if result["refit"].pen_log_lik > best_pen_log_lik:
            best_pen_log_lik = result["refit"].pen_log_lik
            best_worker = succ_worker
    logging.info("Best worker %s", best_worker.out_model_file)
    return best_worker

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        args.num_barcodes = bcode_meta.num_barcodes

    all_topology_files = glob.glob(args.topology_file_template.replace("parsimony_tree0", "parsimony_tree*[0-9]"))
    topology_files = all_topology_files[:args.max_topologies]
    logging.info("Processing the tree files: %s", topology_files)
    best_pen_param, warm_start_files = tune_hyperparams(topology_files, args)
    best_worker = fit_models(topology_files, warm_start_files, args, best_pen_param)

    with open(best_worker.out_model_file, "rb") as f:
        results = six.moves.cPickle.load(f)
    with open(args.out_model_file, "wb") as f:
         six.moves.cPickle.dump(results, f, protocol = 2)
    logging.info("Complete!!!")

if __name__ == "__main__":
    main()
