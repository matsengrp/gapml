#!/bin/bash
#
#SBATCH --job-name=test_gestalt
#SBATCH --output=res_gestalt.txt
# This is an installation script to get things going...

cd /home/jfeng2/gestaltamania/
module load python3/3.4.3
module load buildenv
virtualenv venv_beagle_py3
source venv_beagle_py3/bin/activate
cd gestalt
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

srun "$@"
