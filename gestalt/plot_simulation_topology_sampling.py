import os
import json
import six
import numpy as np

import plot_simulation_common

np.random.seed(0)

seeds = range(211,212)
n_bcode = 1
sampling_rates = [1, 2, 6]
model_seed = 153
lambda_known = 0
tree_idx = 1
do_plots = False
prefix = ""

TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_score.pkl"
RAND_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/collapsed_tree.pkl"
OUT_TRUE_MRCA_PLOT = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_mrca.png"
OUT_FITTED_MRCA_PLOT = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_mrca.png"

def get_true_model(seed, sampling_rate, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, sampling_rate)
    print(file_name)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, sampling_rate, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, sampling_rate, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        sampling_rates,
        n_bcode = n_bcode,
        tree_idx = tree_idx,
        do_plots = do_plots,
        print_keys = [
            "bhv",
            "random_bhv",
            "zero_bhv",
            "super_zero_bhv",
            "mrca",
            "zero_mrca",
            "random_mrca",
            "targ",
            "double"])
