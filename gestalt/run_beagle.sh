#!/bin/bash
#
#SBATCH --job-name=test_gestalt
#SBATCH --output=res_gestalt.txt
#SBATCH -c 18
#SBATCH -N 1
#SBATCH --mem 55000
#SBATCH --exclusive
#SBATCH -p largenode
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeanfeng@uw.edu

cd /home/jfeng2/gestaltamania/
module load python3/3.4.3
module load Java/1.7.0_15
source venv_beagle_py3/bin/activate
cd gestalt

"$@"
