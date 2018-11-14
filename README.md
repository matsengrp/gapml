# Installing things!
You will need to have PHYLIP mix installed so that you can call `mix` on the command line.
http://evolution.genetics.washington.edu/phylip/

We use `pip` to install things into a python virtual environment.
FIRST: if you want tree visualizations from ete3, you will need to do something special since we are using pip (rather than conda).
Install PyQt5 (`sudo apt-get` on linux, `brew install` on mac. For mac, see https://stackoverflow.com/questions/39821177/python-pyqt-on-macos-sierra).
Then run `pip install PyQt5` in your virtual environment.
AFTERWARDS: Install the libraries into your virtual environment through the `requirements.txt` file.

We use nestly + SCons to run simulations/analyses.
You should install scons outside the virtual environment, for a python 2.\* or a 3.5+ environment.
Then activate the virtual environment and then run `scons ___`.

You need to create a file `constant_paths.py` in the `gestalt` folder that provides the paths to the different executables. (mix and bhv distance calculators)

# Running tests
To run all the tests:
```
python3 -m unittest
```
To run specific test module tests/test\_me.py:
```
python3 -m unittest tests.<test_me>
```

# GESTALT pipeline

`generate_data.py`: make data

`restrict_observed_barcodes.py`: restrict to observing the first K alleles

`get_parsimony_trees.py` or `get_collapsed_oracle.py`: create tree topologies to fit to

`tune_topology.py`: to select the best topology given a set of possible topologies

# Fitting the trees via alternative methods
`fit_chronos.py`: uses Sanderson 2002, the `chronos` function in R package `ape`


# Submitting jobs to AWS Batch
We rely on the `boto_run.py` script.
We still run `scons` to submit to AWS batch.
However to ensure dependencies between jobs are upheld, we run each nested command in scons one-by-one.
So we run all `generate_data.py` commands for all seeds first by running `scons` but commenting out the other commands.
Then we run all `restrict_observed_barcodes.py` commands by setting the `generate_data.py` command to an empty command. Then we run `scons`.
This is really hacky but it works for now.
