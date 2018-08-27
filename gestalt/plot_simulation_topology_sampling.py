import os
import json
import six
import numpy as np

import plot_simulation_common
import split_data
from tree_distance import *
from common import assign_rand_tree_lengths

np.random.seed(0)

seeds = range(211,216)
n_bcode = 1
sampling_rates = [1, 2, 6]
model_seed = 153
lambda_known = 0
tree_idx = 1
do_plots = False
prefix = ""
num_rands = 10

TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_score.pkl"
RAND_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_model.pkl"
OUT_TRUE_MRCA_PLOT = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_mrca.png"
OUT_FITTED_MRCA_PLOT = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted_mrca.png"

def get_true_model(seed, sampling_rate, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, sampling_rate)
    return plot_simulation_common.get_true_model(file_name, None, n_bcodes)

def get_result(seed, sampling_rate, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, sampling_rate, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, sampling_rate, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

#plot_simulation_common.gather_results(
#        get_true_model,
#        get_result,
#        get_rand_tree,
#        seeds,
#        sampling_rates,
#        n_bcode = n_bcode,
#        tree_idx = tree_idx,
#        do_plots = do_plots,
#        print_keys = [
#            "bhv",
#            "random_bhv",
#            #"zero_bhv",
#            "super_zero_bhv",
#            "internal_corr",
#            "internal_random_corr",
#            "targ",
#            #"double",
#        ])

"""
Compare only the nodes that were contained in the 10% sample
"""
metric_list = [
        "bhv",
        "random_bhv",
        "zero_bhv",
        "super_zero_bhv",
        "internal_corr",
        "internal_random_corr",
        "leaves"]

n_bcode_results = {
        key: [[] for _ in sampling_rates]
        for key in metric_list}

for seed_idx, seed in enumerate(seeds):
    try:
        true_model = get_true_model(seed, sampling_rates[0], n_bcode)
        tot_height = 1.0 #true_model[0]["time"]
    except FileNotFoundError:
        continue
    comparison_leaves = [leaf.allele_events_list_str for leaf in true_model[tree_idx]]
    true_bhv_meas = BHVDistanceMeasurer(true_model[tree_idx], "_output/scratch")
    true_internal_meas = InternalCorrMeasurer(true_model[tree_idx], "_output/scratch")

    for idx, sampling_rate in enumerate(sampling_rates):
        try:
            result = get_result(seed, sampling_rate, n_bcode)
            result[tree_idx].label_node_ids()
            keep_idxs = []
            for leaf in result[tree_idx]:
                if leaf.allele_events_list_str in comparison_leaves:
                    keep_idxs.append(leaf.node_id)
            result_tree = split_data._prune_tree(result[tree_idx], set(keep_idxs))
        except FileNotFoundError:
            continue

        try:
            rand_res = get_rand_tree(seed, sampling_rate, n_bcode)
            rand_res[tree_idx].label_node_ids()
            keep_idxs = []
            for leaf in rand_res[tree_idx]:
                if leaf.allele_events_list_str in comparison_leaves:
                    keep_idxs.append(leaf.node_id)
            rand_tree = split_data._prune_tree(rand_res[tree_idx], set(keep_idxs))
        except FileNotFoundError:
            continue

        rand_dists = []
        rand_bhv_dists = []
        for _ in range(num_rands):
            assign_rand_tree_lengths(rand_tree, tot_height)
            dist = true_internal_meas.get_dist(rand_tree)
            rand_dists.append(dist)
            dist = true_bhv_meas.get_dist(rand_tree)
            rand_bhv_dists.append(dist)
        rand_dist = np.mean(rand_dists)
        n_bcode_results["random_bhv"][idx].append(np.mean(rand_bhv_dists))
        n_bcode_results["internal_random_corr"][idx].append(np.mean(rand_dists))

        zero_tree = rand_tree.copy()
        for node in zero_tree.traverse():
            if node.is_root():
                continue
            if node.is_leaf():
                node.dist = tot_height - node.up.get_distance(zero_tree)
                assert node.dist > 0
            else:
                node.dist = 1e-10
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["zero_bhv"][idx].append(dist)

        for node in zero_tree:
            node.dist = 0
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["super_zero_bhv"][idx].append(dist)

        dist = true_bhv_meas.get_dist(result_tree)
        n_bcode_results["bhv"][idx].append(dist)
        dist = true_internal_meas.get_dist(result[tree_idx])
        n_bcode_results["internal_corr"][idx].append(dist)
        n_bcode_results["leaves"][idx].append(-1)

for bhvs in n_bcode_results["bhv"]:
    print(bhvs)

plot_simulation_common.print_results(
        sampling_rates,
        n_bcode_results,
        n_bcode,
        print_keys = [
            "bhv",
            "random_bhv",
            "super_zero_bhv",
            "internal_corr",
            "internal_random_corr",
        ])
