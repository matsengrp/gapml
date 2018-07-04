import subprocess
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
            is_refit: bool,
            max_sum_states: int,
            max_extra_steps: int):
        self.obs_file = obs_file
        self.topology_file = topology_file
        self.true_model_file = true_model_file
        self.true_collapsed_tree_file = true_collapsed_tree_file
        self.seed = seed
        self.log_barr = log_barr
        self.target_lam_pen = target_lam_pen
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.is_refit = is_refit
        self.max_sum_states = max_sum_states
        self.max_extra_steps = max_extra_steps

    def run_worker(self, shared_obj):
        """
        @param shared_obj: ignored
        """
        cmd = [
            'python run_estimator.py',
            '--obs-file',
            self.obs_file,
            '--topology-file',
            self.topology_file,
            #'--true-model-file',
            #'--true-collapsed-tree-file',
            '--seed',
            self.seed,
            '--target-lam-pen',
            self.target_lam_pen,
            '--log-barr',
            self.log_barr
            '--max-sum-states',
            self.max_sum_states,
            '--max-iters',
            self.max_iters
        ]
        subprocess.check_call(cmd)
