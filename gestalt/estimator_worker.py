import subprocess
import os
import json
from typing import List

from parallel_worker import ParallelWorker

class RunEstimatorWorker(ParallelWorker):
    def __init__(self,
            obs_file: str,
            topology_file: str,
            true_model_file: str,
            true_collapsed_tree_file: str,
            seed: int,
            log_barr: float,
            target_lam_pen: float,
            max_iters: int,
            num_inits: int,
            do_refit: bool,
            max_sum_states: int,
            max_extra_steps: int,
            scratch_dir: str):
        self.obs_file = obs_file
        self.topology_file = topology_file
        self.out_model_file = topology_file.replace(".pkl", "_fitted.pkl")
        self.out_json_file = self.topology_file.replace(".pkl", "_fitted.json")
        self.true_model_file = true_model_file
        self.true_collapsed_tree_file = true_collapsed_tree_file
        self.seed = seed
        self.log_barr = log_barr
        self.target_lam_pen = target_lam_pen
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.do_refit = do_refit
        self.max_sum_states = max_sum_states
        self.max_extra_steps = max_extra_steps
        self.scratch_dir = scratch_dir

    def run_worker(self, shared_obj):
        """
        @param shared_obj: ignored
        """
        cmd = [
            'python',
            'run_estimator.py',
            '--obs-file',
            self.obs_file,
            '--topology-file',
            self.topology_file,
            '--seed',
            self.seed,
            '--log-barr',
            self.log_barr,
            '--target-lam-pen',
            self.target_lam_pen,
            '--max-iters',
            self.max_iters,
            '--num-inits',
            self.num_inits,
            '--do-refit' if self.do_refit else '',
            '--max-extra-steps',
            self.max_extra_steps,
            '--scratch-dir',
            self.scratch_dir,
        ]
        def _add_more_args(arg_val, arg_key):
            if arg_val is not None:
                return cmd + [arg_key, arg_val]
            else:
                return cmd

        cmd = _add_more_args(
                self.true_model_file,
                '--true-model-file')
        cmd = _add_more_args(
                self.true_collapsed_tree_file,
                '--true-collapsed-tree-file')
        cmd = _add_more_args(
                self.max_sum_states,
                '--max-sum-states')

        subprocess.check_call(list(map(lambda x: str(x), cmd)))

        assert os.path.exists(self.out_json_file)
        with open(self.out_json_file, "r") as f:
            out_dict = json.load(f)
        return out_dict
