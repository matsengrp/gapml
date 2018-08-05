import time
import datetime
import sys
import argparse
import boto3

batch = boto3.client('batch')
jobQueue = "first-run-job-queue"
jobDefinition = "gestaltamania:1"
script = "gestaltamania/gestalt/run_script.sh"
command0 = "/efs/jjfeng/%s" % script
jobNamePrefix = "test_job"

command = [
        # Provide docker container with script name
        command0,
        # Used by run_script.sh to know which folder to cd into
        "jjfeng",
        # Rest of the list are arguments you pass to your script
        "generate_data.py",
        "--is-stupid-cherry",
        "--sampling-rate",
        "1.0",
        "--time",
        "1",
        "--num-barcodes",
        "80",
        "--min-uniq-alleles",
        "3",
        "--max-uniq-alleles",
        "5",
        "--max-abundance",
        "10"
]

command = [
        # Provide docker container with script name
        command0,
        # Used by run_script.sh to know which folder to cd into
        "jjfeng",
        "restrict_observed_barcodes.py",
        "--num-barcodes",
        "80"]

command = [
        command0,
        "jjfeng",
        "get_collapsed_oracle.py",
        "--out-template-file",
        "_output/oracle_tree0.pkl"]

command = [
        command0,
        "jjfeng",
        "run_estimator.py",
        "--topology-file",
        "_output/oracle_tree0.pkl",
        "--max-sum-states",
        "2000",
        "--max-extra-steps",
        "2",
        "--max-iters",
        "10000",
        "--target",
        "1",
        "--num-inits",
        "3"]

resp = batch.submit_job(
        jobName="%s-test" % jobNamePrefix,
        jobQueue=jobQueue,
        jobDefinition=jobDefinition,
        containerOverrides = {
            "command": command,
            "vcpus": 36,
            "memory": 20000
        }
        )
print("Job ID %s" % resp['jobId'])
