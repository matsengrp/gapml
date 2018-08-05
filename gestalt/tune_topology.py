"""
Tunes over different multifurcating topologies.
Basically a wrapper around `run_estimator.py`
This will create jobs to run on the cluster for each of the trees
"""
import sys
import os
import glob
import argparse
import logging
import six

from parallel_worker import BatchSubmissionManager
from estimator_worker import RunEstimatorWorker
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
        '--target-lam-pens',
        type=str,
        default='10',
        help="comma-separated string with penalty parameters on the target lambdas")
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.75,
        help="fraction of data for training data. for tuning penalty param")
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

    parser.set_defaults()
    args = parser.parse_args()

    assert args.log_barr >= 0

    create_directory(args.out_model_file)
    args.topology_folder = os.path.dirname(args.topology_file_template)
    args.scratch_dir = os.path.join(
            args.topology_folder,
            'scratch')
    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    topology_files = glob.glob(args.topology_file_template.replace("parsimony_tree0", "parsimony_tree*[0-9]"))
    logging.info("Processing the tree files: %s", topology_files)
    worker_list = []
    for top_file in topology_files:
        worker = RunEstimatorWorker(
            args.obs_file,
            top_file,
            args.true_model_file,
            args.true_collapsed_tree_file,
            args.seed,
            args.log_barr,
            args.target_lam_pens,
            args.max_iters,
            args.num_inits,
            True, # do refitting
            args.max_sum_states,
            args.max_extra_steps,
            args.train_split,
            args.scratch_dir)
        worker_list.append(worker)

    if len(worker_list) > 1 and args.submit_srun:
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

    best_log_lik = successful_workers[0][0]["log_lik"]
    best_worker = successful_workers[0][1]
    for (result_dict, succ_worker) in successful_workers[1:]:
        if result_dict["log_lik"] > best_log_lik:
            best_log_lik = result_dict["log_lik"]
            best_worker = succ_worker

    logging.info("Best worker %s", best_worker.out_model_file)
    with open(best_worker.out_model_file, "rb") as f:
        results = six.moves.cPickle.load(f)
    with open(args.out_model_file, "wb") as f:
        six.moves.cPickle.dump(results["refit"], f, protocol = 2)
    logging.info("Complete!!!")

if __name__ == "__main__":
    main()
