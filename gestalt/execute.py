"""
Execute commands locally or via sbatch.
"""

import click
import os
import subprocess
import time

sbatch_prelude = """#!/bin/bash
#SBATCH -o %s/job.out
#SBATCH -e %s/job.err
#SBATCH -c 18
#SBATCH -N 1
#SBATCH --mem 55000
#SBATCH --exclusive
#SBATCH -p largenode
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeanfeng@uw.edu

cd /home/jfeng2/gestaltamania/gestalt
module load python3/3.4.3
module load Java/1.7.0_15
source ../venv_beagle_py3/bin/activate
"""


@click.command()
@click.option('--clusters', default='', help='Clusters to submit to. Default is local execution.')
@click.option('--max-tries', default=2, help='Maximum number of times to try resubmitting to cluster')
@click.argument('target')
@click.argument('to_execute_str')
def cli(clusters, max_tries, target, to_execute_str):
    """
    Execute a command with targets, perhaps on a SLURM
    cluster via sbatch. Wait until the command has completed.

    TARGETS: Output files as a space-separated list.

    TO_EXECUTE_F_STRING: The command to execute
    """
    if clusters == '':
        # Local execution.
        click.echo("Executing locally:")
        click.echo(to_execute_str)
        return subprocess.check_output(to_execute_str, shell=True)

    # Put the batch script in the directory of the first target.
    execution_dir = os.path.dirname(target)
    script_name = 'job.sh'
    script_full_path = os.path.join(execution_dir, script_name)
    sentinel_path = os.path.join(execution_dir, 'sentinel.txt')
    with open(script_full_path, 'w') as fp:
        fp.write(sbatch_prelude % (execution_dir, execution_dir))
        fp.write(to_execute_str + '\n')
        fp.write('touch %s\n' % sentinel_path)

    for _ in range(max_tries):
        out = subprocess.check_output(
                'sbatch --clusters %s %s' % (clusters, script_full_path),
                shell=True)
        click.echo(out.decode('UTF-8'))

        # Wait until the sentinel file appears, then clean up.
        while not os.path.exists(sentinel_path):
            time.sleep(5)
        os.remove(sentinel_path)

        # Check if the job failed some weird death
        # This is caused when I import tensorflow_probability
        # I think it has to do with a bad machine + tensorflow_probability.
        # the current solution is to wait long enough, resubmit a new job, and then
        # hope we got a new machine that won't cry
        with open(os.path.join(execution_dir, "job.err"), "r") as err_f:
            do_retry = any(["Illegal instruction" in line for line in err_f.readlines()])
        if not do_retry:
            break
        else:
            # Wait 1000 seconds and hope we can get hold of a new machine
            print("Sadness. Waiting before we resubmit...")
            time.sleep(1000)

    return int(do_retry)


if __name__ == '__main__':
    cli()
