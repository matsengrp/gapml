import sys
import os
import shutil
import traceback
import six
import time
import logging
import custom_utils
from custom_utils import CustomCommand, run_cmd, finish_process
import numpy as np


class BatchParallelWorkers:
    def __init__(self, workers, shared_obj):
        self.workers = workers
        self.shared_obj = shared_obj


class ParallelWorker:
    """
    Stores the information for running something in parallel
    These workers can be run throught the ParallelWorkerManager
    """
    def __init__(self, seed):
        """
        @param seed: a seed for for each parallel worker
        """
        raise NotImplementedError()

    def run(self, shared_obj):
        """
        @param shared_obj: an object that is taken in - shared among ParallelWorkers

        Do not implement this function!
        """
        np.random.seed(self.seed)

        result = None
        try:
            result = self.run_worker(shared_obj)
        except Exception as e:
            print("Exception caught in parallel worker: %s" % e)
            traceback.print_exc()
        return result

    def run_worker(self, shared_obj):
        """
        @param shared_obj: an object that is taken in - shared among ParallelWorkers

        Implement this function!
        Returns whatever value needed from this task
        """
        raise NotImplementedError()


class ParallelWorkerManager:
    """
    Runs many ParallelWorkers
    """
    def run(self):
        raise NotImplementedError()

    def create_batch_worker_cmds(self, worker_list, num_approx_batches):
        """
        Create commands for submitting to a batch manager
        Pickles the workers as input files to the jobs
        The commands specify the output file names for each job - read these output files
        to retrieve the results from the jobs
        """
        num_workers = len(worker_list)
        num_per_batch = int(max(np.ceil(float(num_workers)/num_approx_batches), 1))
        for batch_idx, start_idx in enumerate(range(0, num_workers, num_per_batch)):
            batched_workers = worker_list[start_idx:start_idx + num_per_batch]
            self.batched_workers.append(batched_workers)

            # Create the folder for the output from this batch worker
            worker_batch_folder = "%s/batch_%d" % (self.worker_folder, batch_idx)
            if not os.path.exists(worker_batch_folder):
                os.makedirs(worker_batch_folder)
            self.output_folders.append(worker_batch_folder)

            # Create the command for this batch worker
            input_file_name = "%s/in.pkl" % worker_batch_folder
            output_file_name = "%s/out.pkl" % worker_batch_folder
            log_file_name = "%s/log.txt" % worker_batch_folder
            self.output_files.append(output_file_name)
            with open(input_file_name, "wb") as cmd_input_file:
                # Pickle the worker as input to the job
                six.moves.cPickle.dump(
                    BatchParallelWorkers(batched_workers, self.shared_obj),
                    cmd_input_file,
                    protocol=2,
                )
                cmd_str = "python run_worker.py --input-file %s --output-file %s --log-file %s" % (input_file_name, output_file_name, log_file_name)
                print(cmd_str)
                batch_cmd = CustomCommand(
                    cmd_str,
                    outfname=output_file_name,
                    logdir=worker_batch_folder,
                    env=os.environ.copy(),
                )
                self.batch_worker_cmds.append(batch_cmd)

    def read_batch_worker_results(self):
        """
        Read the output (pickle) files from the batched workers
        """
        worker_results = []
        for i, f in enumerate(self.output_files):
            try:
                with open(f, "rb") as output_f:
                    res = six.moves.cPickle.load(output_f)
            except Exception as e:
                # Probably the file doesn't exist and the job failed?
                traceback.print_exc()
                if self.retry:
                    print("Rerunning locally -- could not load pickle files %s" % f)
                    # Now let's try to recover by running the worker
                    res = [w.run(self.shared_obj) for w in self.batched_workers[i]]

            for j, r in enumerate(res):
                if r is None:
                    print("WARNING: batch submission manager, worker failed %s" % self.batched_workers[i][j].name)
                    worker_results.append(None)
                else:
                    worker_results.append(r)

        return worker_results

    def clean_outputs(self):
        try:
            for fname in self.output_folders:
                if os.path.exists(fname):
                    shutil.rmtree(fname)
        except Exception as e:
            logging.info("failed to clean: %s", e)

    def _get_successful_jobs(self, results, workers):
        """
        filters our the results that were not successful
        @return list of tuples with (result, worker)
        """
        successful_res_workers = []
        for res, worker in zip(results, workers):
            if res is None:
                continue
            successful_res_workers.append((res, worker))
        return successful_res_workers


class SubprocessManager(ParallelWorkerManager):
    """
    Creates separate processes on the same CPU to run the workers
    """
    def __init__(
            self,
            worker_list,
            shared_obj,
            worker_folder,
            num_processes,
            retry=False):
        self.batch_worker_cmds = []
        self.batched_workers = [] # Tracks the batched workers if something fails
        self.output_folders = []
        self.output_files = []

        self.retry = retry
        self.worker_list = worker_list
        self.worker_folder = worker_folder
        self.num_processes = num_processes
        self.shared_obj = shared_obj
        self.create_batch_worker_cmds(worker_list, len(worker_list))
        self.batch_system = "subprocess"

    def run(self, successful_only=False, sleep=0.01):
        """
        @param successful_only: whether to return successful jobs only
                                unsuccessful jobs have None as their result
        @return list of tuples (result, worker)
        """
        cmdfos = self.batch_worker_cmds
        procs = []
        n_tries = []
        up_to_idx = 0

        def get_num_unused_num_processes(procs):
            return self.num_processes - (len(procs) - procs.count(None))

        while up_to_idx < len(cmdfos) - 1 or len(cmdfos) != procs.count(None):
            num_unused_num_processes = get_num_unused_num_processes(procs)
            if num_unused_num_processes > 0 and up_to_idx <= len(cmdfos):
                old_idx = up_to_idx
                up_to_idx += num_unused_num_processes
                for cmd in cmdfos[old_idx:up_to_idx]:
                    procs.append(run_cmd(
                        cmd,
                        batch_system=self.batch_system))
                    n_tries.append(1)
                    if sleep:
                        time.sleep(sleep)

            # we set each proc to None when it finishes
            for iproc in range(len(procs)):
                if procs[iproc] is None:  # already finished
                    continue
                if procs[iproc].poll() is not None:  # it just finished
                    finish_process(
                            iproc,
                            procs,
                            n_tries,
                            cmdfos[iproc],
                            batch_system=self.batch_system,
                            max_num_tries=0)
            sys.stdout.flush()
            if sleep:
                time.sleep(sleep)

        res = self.read_batch_worker_results()
        self.clean_outputs()
        if successful_only:
            return self._get_successful_jobs(res, self.worker_list)
        else:
            return [(r, w) for r, w in zip(res, self.worker_list)]
