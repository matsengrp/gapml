import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

seeds = range(210,211)
n_bcode = 1
sampling_rates = [1, 2, 6]
model_seed = 153
lambda_known = 1
prefix = ""

TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/lambda_known%d/tot_time_known1/tune_fitted.pkl"
TRUE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topology_sampling/_output/model_seed%d/%d/sampling%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, lambda_type)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, lambda_type, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, lambda_type, n_bcodes, lambda_known)
    return plot_simulation_common.get_result(res_file)

get_param_func_dict = {
        "mrca": None, # custom function
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
        for idx, lambda_type in enumerate(sampling_rates):
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
    try:
        true_model = get_true_model(seed, sampling_rates[0], n_bcode)
        print("123123123")
    except FileNotFoundError:
        print("asdfasdf")
        continue
    true_subtree = true_model[1]
    sampled_node_ids = set()
    for leaf in true_subtree:
        sampled_node_ids.add(leaf.node_id)
        leaf.add_feature("sampled_node_id", leaf.node_id)

    true_mrca_meas = MRCADistanceMeasurer(true_subtree, attr="sampled_node_id")
    #print(true_subtree.get_ascii(attributes=["dist"], show_internal=True))

    for idx, sampling_rate in enumerate(sampling_rates):
        try:
            true_model = get_true_model(seed, sampling_rate, n_bcode)
            allele_to_node_id_dict = {}
            for leaf in true_model[1]:
                allele_to_node_id_dict[leaf.allele_events_list_str] = leaf.node_id
        except FileNotFoundError:
            print("asdfasdf 2222")
            continue
        n_bcode_results["leaves"][idx].append(len(true_model[2]))

        try:
            result = get_result(seed, sampling_rate, n_bcode)
            keep_leaves = []
            for leaf in result[1]:
                orig_node_id = allele_to_node_id_dict[leaf.allele_events_list_str]
                if orig_node_id in sampled_node_ids:
                    leaf.add_feature(
                        "sampled_node_id",
                        allele_to_node_id_dict[leaf.allele_events_list_str])
                    keep_leaves.append(leaf)
            tot_dist = keep_leaves[0].get_distance(result[1])
            result[1].prune(keep_leaves)
            pruned_tree = result[1]
            # ETE pruning doesnt really work... need to fix distances it seems
            for leaf in pruned_tree:
                if leaf.dist == 0:
                    leaf.dist = tot_dist - leaf.get_distance(pruned_tree)
        except FileNotFoundError:
            print("asdfasdf11111")
            continue
        n_bcode_results["seed"][idx].append(seed)
        #print(pruned_tree.get_ascii(attributes=["dist"], show_internal=True))
        dist = true_mrca_meas.get_dist(pruned_tree)
        n_bcode_results["mrca"][idx].append(dist)

        true_bhv_meas = MRCASpearmanMeasurer(true_subtree, "_output/scratch", attr="sampled_node_id")
        dist = true_bhv_meas.get_dist(pruned_tree)
        #true_bhv_meas = MRCASpearmanMeasurer(true_subtree, attr="sampled_node_id")
        #dist = true_bhv_meas.get_dist(pruned_tree)
        n_bcode_results["tau"][idx].append(dist)

plot_simulation_common.print_results(sampling_rates, n_bcode_results, n_bcode)
