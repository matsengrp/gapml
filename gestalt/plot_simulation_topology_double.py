import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

model_seed = 500
seeds = range(900,905)
double_cuts = [1, 6, 36]
n_bcode = 1
prefix = ""
lambda_known = 0
tree_idx = 1
do_plots = False

TEMPLATE = "%ssimulation_topology_double/_output/model_seed%d/%d/double_cut%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted.pkl"
RAND_TEMPLATE = "%ssimulation_topology_double/_output/model_seed%d/%d/double_cut%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_double/_output/model_seed%d/%d/double_cut%d/true_model.pkl"
OBS_TEMPLATE = "%ssimulation_topology_double/_output/model_seed%d/%d/double_cut%d/num_barcodes%d/obs_data.pkl"

def get_true_model(seed, double_cut, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, double_cut)
    return plot_simulation_common.get_true_model(file_name, None, n_bcodes)

def get_result(seed, double_cut, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, double_cut, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, double_cut, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, double_cut, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        double_cuts,
        n_bcode = n_bcode,
        tree_idx = tree_idx,
        do_plots = do_plots,
        print_keys = [
            "bhv",
            "random_bhv",
            #"zero_bhv",
            "super_zero_bhv",
            #"mrca",
            #"zero_mrca",
            #"random_mrca",
            "targ",
            "double",
       ])
