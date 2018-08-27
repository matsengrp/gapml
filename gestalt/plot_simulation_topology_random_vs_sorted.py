import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

seeds = range(10,18)
n_bcode = 1
seq_idxs = range(5)
model_seed = 900
lambda_known = 0
prefix = ""
tree_idx = 1
do_plots = False
num_rands = 10

TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/seq_idx%d/data_seed%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_score.pkl"
RAND_TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/seq_idx%d/data_seed%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/seq_idx%d/data_seed%d/true_model.pkl"

def get_true_model(seed, seq_idx, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seq_idx, seed)
    return plot_simulation_common.get_true_model(file_name, None, n_bcodes)

def get_result(seed, seq_idx, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seq_idx, seed, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, seq_idx, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seq_idx, seed, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        seq_idxs,
        n_bcode = n_bcode,
        tree_idx = tree_idx,
        do_plots = do_plots,
        print_keys = [
            "bhv",
            "random_bhv",
            #"zero_bhv",
            "super_zero_bhv",
            "internal_corr",
            "internal_random_corr",
            "targ",
            #"double",
        ])
