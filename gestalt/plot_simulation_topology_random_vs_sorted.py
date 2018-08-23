import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

seeds = range(700,705)
n_bcode = 1
lambda_types = ["random", "sorted"]
model_seed = 903
lambda_known = 0
prefix = ""
tree_idx = 1
do_plots = False
prefix = ""
num_rands = 10

TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/%d/%s/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted.pkl"
RAND_TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/%d/%s/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/%d/%s/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topology_random_vs_sorted/_output/model_seed%d/%d/%s/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, lambda_type)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, lambda_type, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, lambda_type, n_bcodes, lambda_known)
    print(res_file)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, lambda_type, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, lambda_type, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        lambda_types,
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
