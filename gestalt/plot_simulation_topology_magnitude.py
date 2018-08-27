import os
import json
import six
import numpy as np

import plot_simulation_common

np.random.seed(0)

double = 0
model_seed = 90
seeds = range(150,155)
n_bcode = 1
lambda_magnitudes = [3, 10, 30]
prefix = ""
lambda_known = 0
tree_idx = 1
do_plots = False

TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_score.pkl"
RAND_TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/true_model.pkl"
OUT_TRUE_MRCA_PLOT = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/true_mrca.png"
OUT_FITTED_MRCA_PLOT = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_mrca.png"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, lambda_type, double)
    return plot_simulation_common.get_true_model(file_name, None, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, lambda_type, double, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, lambda_type, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        lambda_magnitudes,
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
            #"double",
       ])
