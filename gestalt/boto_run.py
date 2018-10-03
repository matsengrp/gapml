import time
import datetime
import sys
import argparse
import boto3
import argparse

batch = boto3.client('batch')
jobQueue = "first-run-job-queue"
jobDefinition = "gestaltamania:2"
script = "gestaltamania/gestalt/run_script.sh"
command0 = "/efs/jjfeng/%s" % script
jobNamePrefix = "test_job"

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
    parser.add_argument(
        '--vcpus',
        type=int,
        default=2)
    parser.add_argument(
        '--memory',
        type=int,
        default=2000,
        help='mem in MB')
    parser.add_argument(
        '--cmd',
        type=str,
        help="command to run on AWS")
    parser.add_argument(
        '--in-job-dependency-file',
        type=str,
        help="input file with job dependency ids")
    parser.add_argument(
        '--out-job-id',
        type=str,
        help="file with job id")
    parser.add_argument(
        '--debug',
        action='store_true',
        help="dont actually submit")

    parser.set_defaults()
    args = parser.parse_args(args)
    assert args.vcpus >= 2
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    command = [command0, "jjfeng"] + args.cmd.split(" ")
    print(command)
    job_ids = []
    if args.in_job_dependency_file:
        with open(args.in_job_dependency_file, "r") as f:
            job_id = f.readline()
            job_ids = [job_id]
    print("depends on", job_ids)
    if not args.debug:
        resp = batch.submit_job(
            jobName="%s-test" % jobNamePrefix,
            jobQueue=jobQueue,
            jobDefinition=jobDefinition,
            dependsOn=[{
                'jobId': job_id,
            } for job_id in job_ids],
            containerOverrides={
                "command": command,
                "vcpus": args.vcpus,
                "memory": args.memory
            })
        print("New Job ID %s" % resp['jobId'])
        print("out job file", args.out_job_id)
        with open(args.out_job_id, "w") as f:
            f.write("%s" % resp['jobId'])

if __name__ == "__main__":
    main()
