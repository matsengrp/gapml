#!/bin/bash
#
#SBATCH --job-name=test_gestalt
#SBATCH --output=res_gestalt.txt
# This is an installation script to get things going...

cd /home/jfeng2/gestaltamania/
module load python3/3.4.3
module load buildenv
# Use a different virtual env for old vs. new beagle machines
virtualenv venv_beagle_py3_old
source venv_beagle_py3_old/bin/activate
cd gestalt
pip install --upgrade pip
pip install -r requirements_beagle_old_machines.txt --force-reinstall
deactivate

# Use this one for new machines
virtualenv venv_beagle_py3
source venv_beagle_py3/bin/activate
cd gestalt
pip install --upgrade pip
pip install -r requirements_beagle_new_machines.txt --force-reinstall
deactivate
