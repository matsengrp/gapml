# Installing things!
You will need to have PHYLIP mix installed so that you can call `mix` on the command line.

We use `pip` to install things into a python virtual environment.
Install the libraries into your virtual environment through the `requirements.txt` file.
Note: to get tree visualizations from ete3, you will need to do something special since we are using pip (rather than conda).
Install PyQt5 (`sudo apt-get` on linux, `brew install` on mac. For mac, see https://stackoverflow.com/questions/39821177/python-pyqt-on-macos-sierra).
Then run `pip install PyQt5` in your virtual environment.

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

`run_estimator.py`: fit the estimator for a given topology, run statistics if given the true tree
