import time
import datetime
import sys
import argparse
import boto3
import argparse

batch = boto3.client('batch')
jobQueue = "first-run-job-queue"
jobDefinition = "gestaltamania:1"
script = "gestaltamania/gestalt/run_script.sh"
command0 = "/efs/jjfeng/%s" % script
jobNamePrefix = "test_job"

def main(args=sys.argv[1:]):
    vcpus = int(args[0])
    assert vcpus >= 2
    memory = int(args[1])
    # For run_estimator, choose memory something like 20000
    command = [command0, "jjfeng"] + (" ".join(args[2:])).split(" ")
    print(command)
    resp = batch.submit_job(
            jobName="%s-test" % jobNamePrefix,
            jobQueue=jobQueue,
            jobDefinition=jobDefinition,
            containerOverrides = {
                "command": command,
                "vcpus": vcpus,
                "memory": memory
            }
            )
    print("Job ID %s" % resp['jobId'])

if __name__ == "__main__":
    main()
