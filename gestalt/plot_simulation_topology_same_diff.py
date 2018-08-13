import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

double = 0
seeds = range(0,5)
n_bcode = 1
lambda_types = ["same", "diff", "super_diff"]


TEMPLATE = "simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut%d/num_barcodes%d/lambda_known1/abundance_weight0/tune_fittedtree0.pkl"
TRUE_TEMPLATE = "simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut%d/true_model.pkl"
OBS_TEMPLATE = "simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut%d/num_barcodes%d/obs_data.pkl"
COLL_TREE_TEMPLATE = "simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (seed, lambda_type, double)
    tree_file_name = COLL_TREE_TEMPLATE % (seed, lambda_type, double, n_bcodes)
    print(file_name)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (seed, lambda_type, double, n_bcodes)
    print(res_file)
    return plot_simulation_common.get_result(res_file)

get_param_func_dict = {
        "mrca": None, # custom function
        "bhv": None, # custom function
        "targ": plot_simulation_common.get_target_lams,
        "double": plot_simulation_common.get_double_cut_weight,
        "leaves": None,
        "num_indels": None}

n_bcode_results = {
        key: [[] for _ in lambda_types]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, lambda_type in enumerate(lambda_types):
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
    for idx, lambda_type in enumerate(lambda_types):
        try:
            true_model = get_true_model(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        true_mrca_meas = MRCADistanceMeasurer(true_model[1])
        n_bcode_results["leaves"][idx].append(len(true_model[2]))

        num_indels = 0
        for leaf in true_model[2]:
            num_indels += len(leaf.allele_events_list[0].events)
        n_bcode_results["num_indels"][idx].append(num_indels)
        #print("true...", true_mrca_meas.ref_tree_mrca_matrix.shape)
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        dist = true_mrca_meas.get_dist(result[1])
        n_bcode_results["mrca"][idx].append(dist)

        true_bhv_meas = MRCASpearmanMeasurer(true_model[1], "_output/scratch")
        dist = true_bhv_meas.get_dist(result[1])
        n_bcode_results["bhv"][idx].append(dist)

        #true_bhv_meas = BHVDistanceMeasurer(true_model[1], "_output/scratch")
        #dist = true_bhv_meas.get_dist(result[1])
        #n_bcode_results["bhv"][idx].append(dist)

plot_simulation_common.print_results(lambda_types, n_bcode_results, n_bcode)
