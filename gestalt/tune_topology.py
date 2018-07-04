import sys
import argparse

from parallel_worker import BatchSubmissionManager
from estimator_worker import RunEstimatorWorker

def parse_args():
    parser = argparse.ArgumentParser(description='fit topology and branch lengths for GESTALT')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--out-model-file',
        type=str,
        default="_output/tune_topology_fitted.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log_tune_topology.txt")
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")
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
        '--target-lam-penalties',
        type=str,
        default="10",
        help="comma-separated string with penalty parameters on the target lambdas")
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

    parser.set_defaults()
    args = parser.parse_args()

    assert args.log_barr >= 0
    assert args.target_lam_pen >= 0
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    topology_files = [
            '_output/parsimony_tree0.pkl',
            '_output/parsimony_tree1.pkl',
            ]
    worker_list = []
    for top_file in topology_files:
        worker = RunEstimatorWorker(
            args.obs_file,
            args.topology_file,
            args.true_model_file,
            args.true_collapsed_tree_file,
            args.seed,
            args.log_barr,
            args.target_lam_pen,
            args.max_iters,
            args.num_inits,
            args.is_refit,
            args.max_sum_states,
            args.max_extra_steps)
        worker_list.append(worker)

    job_manager = BatchSubmissionManager(
            worker_list,
            None,
            len(worker_list),
            args.scratch_dir)
    worker_results = job_manager.run()

if __name__ == "__main__":
    main()
