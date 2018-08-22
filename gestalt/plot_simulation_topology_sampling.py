import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
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

get_param_func_dict = {
        "mrca": None, # custom function
        "bhv": None, # custom function
        "random_bhv": None, # custom function
        "zero_bhv": None, # custom function
        "super_zero_bhv": None, # custom function
        "zero_mrca": None, # custom function
        "random_mrca": None, # custom function
        "leaves": None, # custom function
        "seeds": None, # custom function
        "targ": plot_simulation_common.get_only_target_lams,
        "double": plot_simulation_common.get_double_cut}

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

for seed_idx, seed in enumerate(seeds):
    for idx, sampling_rate in enumerate(sampling_rates):
        try:
            true_model = get_true_model(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue
        n_bcode_results["leaves"][idx].append(len(true_model[2]))
        true_mrca_meas = MRCADistanceMeasurer(true_model[tree_idx])
        true_bhv_meas = BHVDistanceMeasurer(true_model[tree_idx], "_output/scratch")

        if seed_idx == 0 and do_plots:
            plot_simulation_common.plot_mrca_matrix(
                true_mrca_meas.ref_tree_mrca_matrix,
                OUT_TRUE_MRCA_PLOT % (prefix, model_seed, seed, sampling_rate))

        try:
            result = get_result(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue
        n_bcode_results["seeds"][idx].append(seed)
        if seed_idx == 0 and do_plots:
            plot_simulation_common.plot_mrca_matrix(
                true_mrca_meas._get_mrca_matrix(result[tree_idx]),
                OUT_FITTED_MRCA_PLOT % (prefix, model_seed, seed, sampling_rate, n_bcode, lambda_known))

        #print(pruned_tree.get_ascii(attributes=["dist"], show_internal=True))
        mle_dist = true_mrca_meas.get_dist(result[tree_idx])
        n_bcode_results["mrca"][idx].append(mle_dist)

        try:
            rand_tree = get_rand_tree(seed, sampling_rate, n_bcode)
        except FileNotFoundError:
            continue

        rand_dists = []
        rand_bhv_dists = []
        for _ in range(10):
            br_scale = 0.8
            has_neg = True
            while has_neg:
                has_neg = False
                for node in rand_tree[tree_idx].traverse():
                    if node.is_root():
                        continue
                    if node.is_leaf():
                        node.dist = 1 - node.up.get_distance(rand_tree[tree_idx])
                        if node.dist < 0:
                            has_neg = True
                            break
                    else:
                        node.dist = np.random.rand() * br_scale
                br_scale *= 0.8
            dist = true_mrca_meas.get_dist(rand_tree[tree_idx])
            rand_dists.append(dist)
            dist = true_bhv_meas.get_dist(rand_tree[tree_idx])
            rand_bhv_dists.append(dist)
        rand_dist = np.mean(rand_dists)
        n_bcode_results["random_bhv"][idx].append(np.mean(rand_bhv_dists))
        n_bcode_results["random_mrca"][idx].append(np.mean(rand_dists))

        zero_tree = rand_tree[tree_idx].copy()
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
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["zero_bhv"][idx].append(dist)

        for node in zero_tree:
            node.dist = 0
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["super_zero_bhv"][idx].append(dist)

        n_bcode_results["seeds"][idx].append(len(true_model[tree_idx]))

        dist = true_mrca_meas.get_dist(result[tree_idx])
        n_bcode_results["mrca"][idx].append(dist)

        dist = true_bhv_meas.get_dist(result[tree_idx])
        n_bcode_results["bhv"][idx].append(dist)

plot_simulation_common.print_results(
        sampling_rates,
        n_bcode_results,
        n_bcode,
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
