import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

double = 0
model_seed = 9
seeds = range(150,155)
n_bcode = 1
lambda_magnitudes = [3, 10, 30]
prefix = "tmp_mount/"

TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/num_barcodes%d/lambda_known1/abundance_weight0/tune_fitted.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topology_magnitude/_output/model_seed%d/%d/lambda_magnitude%d/double_cut%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, lambda_type, double)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_result(res_file)

get_param_func_dict = {
        "mrca": None, # custom function
        "tau": None, # custom function
        "targ": plot_simulation_common.get_target_lams,
        "double": plot_simulation_common.get_double_cut_weight,
        "leaves": None}

n_bcode_results = {
        key: [[] for _ in lambda_magnitudes]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, lambda_type in enumerate(lambda_magnitudes):
            try:
                true_model = get_true_model(seed, lambda_type, n_bcode)
            except FileNotFoundError:
                continue
            true_model_val = get_param_func(true_model)
            try:
                result = get_result(seed, lambda_type, n_bcode)
            except FileNotFoundError:
                continue
            fitted_val = get_param_func(result)
            dist = np.linalg.norm(fitted_val - true_model_val)
            n_bcode_results[key][idx].append(dist)

for seed in seeds:
    for idx, lambda_type in enumerate(lambda_magnitudes):
        try:
            true_model = get_true_model(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        true_mrca_meas = MRCADistanceMeasurer(true_model[1])
        n_bcode_results["leaves"][idx].append(len(true_model[2]))
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        dist = true_mrca_meas.get_dist(result[1])
        n_bcode_results["mrca"][idx].append(dist)

        true_tau_meas = GavruskinMeasurer(true_model[1], "_output/scratch")
        dist = true_tau_meas.get_dist(result[1])
        n_bcode_results["tau"][idx].append(dist)

plot_simulation_common.print_results(lambda_magnitudes, n_bcode_results, n_bcode)
