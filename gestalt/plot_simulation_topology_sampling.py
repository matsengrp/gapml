import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

seeds = range(210,215)
n_bcode = 1
sampling_rates = [1, 2, 6]
model_seed = 153
lambda_known = 0
prefix = ""

TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted.pkl"
RAND_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, sampling_rate, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, sampling_rate)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, sampling_rate, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, sampling_rate, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

get_param_func_dict = {
        "mrca": None, # custom function
        "random_mrca": None, # custom function
        "zero_mrca": None, # custom function
        "tau": None, # custom function
        "targ": plot_simulation_common.get_target_lams,
        "double": plot_simulation_common.get_double_cut_weight,
        "leaves": None,
        "seed": None}

n_bcode_results = {
        key: [[] for _ in sampling_rates]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, sampling_rate in enumerate(sampling_rates):
            try:
                true_model = get_true_model(seed, sampling_rate, n_bcode)
            except FileNotFoundError:
                continue
            true_model_val = get_param_func(true_model)
            try:
                result = get_result(seed, sampling_rate, n_bcode)
            except FileNotFoundError:
                continue
            fitted_val = get_param_func(result)
            dist = np.linalg.norm(fitted_val - true_model_val)
            n_bcode_results[key][idx].append(dist)

for seed in seeds:
    for idx, sampling_rate in enumerate(sampling_rates):
        try:
            true_model = get_true_model(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue
        n_bcode_results["leaves"][idx].append(len(true_model[2]))
        true_mrca_meas = MRCADistanceMeasurer(true_model[1])

        try:
            result = get_result(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue
        n_bcode_results["seed"][idx].append(seed)
        #print(pruned_tree.get_ascii(attributes=["dist"], show_internal=True))
        dist = true_mrca_meas.get_dist(result[1])
        n_bcode_results["mrca"][idx].append(dist)

        true_bhv_meas = MRCASpearmanMeasurer(true_model[1], "_output/scratch")
        dist = true_bhv_meas.get_dist(result[1])
        #true_bhv_meas = MRCASpearmanMeasurer(true_subtree, attr="sampled_node_id")
        #dist = true_bhv_meas.get_dist(pruned_tree)
        n_bcode_results["tau"][idx].append(dist)

        try:
            _, rand_tree = get_rand_tree(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue

        for node in rand_tree.traverse():
            if node.is_root():
                continue
            if node.is_leaf():
                node.dist = 1 - node.up.get_distance(rand_tree)
                assert node.dist > 0
            else:
                node.dist = 0.025
        dist = true_mrca_meas.get_dist(rand_tree)
        n_bcode_results["random_mrca"][idx].append(dist)

        zero_tree = rand_tree.copy()
        for node in zero_tree.traverse():
            if node.is_root():
                continue
            if node.is_leaf():
                node.dist = 1 - node.up.get_distance(zero_tree)
                assert node.dist > 0
            else:
                node.dist = 1e-10
        dist = true_mrca_meas.get_dist(zero_tree)
        n_bcode_results["zero_mrca"][idx].append(dist)

plot_simulation_common.print_results(sampling_rates, n_bcode_results, n_bcode)
