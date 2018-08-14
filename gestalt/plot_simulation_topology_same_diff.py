import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

double = 0
model_seed = 301
seeds = range(0,5)
n_bcode = 1
lambda_types = ["same", "diff", "super_diff"]
lambda_known = 1
mrcaC = 0.1

TEMPLATE = "simulation_topology_same_diff/_output/model_seed%d/%d/%s/double_cut%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted.pkl"
RAND_TEMPLATE = "simulation_topology_same_diff/_output/model_seed%d/%d/%s/double_cut%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "simulation_topology_same_diff/_output/model_seed%d/%d/%s/double_cut%d/true_model.pkl"
COLL_TREE_TEMPLATE = "simulation_topology_same_diff/_output/model_seed%d/%d/%s/double_cut%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (model_seed, seed, lambda_type, double)
    tree_file_name = COLL_TREE_TEMPLATE % (model_seed, seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (model_seed, seed, lambda_type, double, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, lambda_type, n_bcodes):
    res_file = RAND_TEMPLATE % (model_seed, seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

get_param_func_dict = {
        "zero_mrca": None, # custom function
        "random_mrca": None, # custom function
        "mrca": None, # custom function
        "tau": None, # custom function
        "targ": plot_simulation_common.get_target_lams,
        "leaves": None,
        "seeds": None,
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
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            print("not found")
            continue
        n_bcode_results["seeds"][idx].append(seed)

        dist = true_mrca_meas.get_dist(result[1], C=mrcaC)
        n_bcode_results["mrca"][idx].append(dist)

        try:
            _, rand_tree = get_rand_tree(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue

        rand_dists = []
        for _ in range(10):
            br_scale = 0.8
            has_neg = True
            while has_neg:
                has_neg = False
                for node in rand_tree.traverse():
                    if node.is_root():
                        continue
                    if node.is_leaf():
                        node.dist = 1 - node.up.get_distance(rand_tree)
                        if node.dist < 0:
                            has_neg = True
                            break
                    else:
                        node.dist = np.random.rand() * br_scale
                br_scale *= 0.8
            dist = true_mrca_meas.get_dist(rand_tree, C=mrcaC)
            rand_dists.append(dist)
        n_bcode_results["random_mrca"][idx].append(np.mean(rand_dists))

        zero_tree = rand_tree.copy()
        for node in zero_tree.traverse():
            if node.is_root():
                continue
            if node.is_leaf():
                node.dist = 1 - node.up.get_distance(zero_tree)
                assert node.dist > 0
            else:
                node.dist = 1e-10
        dist = true_mrca_meas.get_dist(zero_tree, C=mrcaC)
        n_bcode_results["zero_mrca"][idx].append(dist)

        #true_bhv_meas = BHVDistanceMeasurer(true_model[1], "_output/scratch")
        #dist = true_bhv_meas.get_dist(result[1])
        #n_bcode_results["bhv"][idx].append(dist)

plot_simulation_common.print_results(
        lambda_types,
        n_bcode_results,
        n_bcode,
        print_keys = [
            "mrca",
            "zero_mrca",
            "random_mrca",
            "targ"])
