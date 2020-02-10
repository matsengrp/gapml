# GAPML: GESTALT analysis using penalized maximum likelihood
Associated paper: Feng et al, "Estimation of cell lineage trees by maximum-likelihood phylogenetics" (submitted for review), preprint available on [BioRxiv](https://www.biorxiv.org/content/10.1101/595215v1).

# Installation
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

# GESTALT pipeline

`generate_data.py`: make data

`restrict_observed_barcodes.py`: restrict to observing the first K alleles

`get_parsimony_topologies.py` or `get_collapsed_oracle.py`: create tree topologies to fit to

`tune_topology.py`: to select the best topology given a set of possible topologies

`convert_to_newick.py`: output the fitted tree in newick format

# Fitting the trees via alternative methods
`fit_chronos.py`: uses Sanderson 2002, the `chronos` function in R package `ape`

# Running analyses in the paper
We do this using scons.
For example: `scons simulation_topol_consist/ -n --clusters=beagle`

# Running tests
To run all the tests:
```
python3 -m unittest
```
To run specific test module tests/test\_me.py:
```
python3 -m unittest tests.<test_me>
```
