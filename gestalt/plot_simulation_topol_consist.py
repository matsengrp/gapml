import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

model_seed = 394
seeds = range(500,501)
num_barcodes = [1, 4, 8, 40]
prefix = ""
tree_idx = 1

TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_fitted.pkl"
RAND_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/collapsed_tree.pkl"
OUT_TRUE_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/true_mrca.png"
OUT_FITTED_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_fitted_mrca.png"
OUT_RAND_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_rand_mrca.png"

def get_true_model(seed, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    print(res_file)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    print(res_file)
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
        key: [[] for _ in num_barcodes]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, n_bcode in enumerate(num_barcodes):
            try:
                true_model = get_true_model(seed, n_bcode)
            except FileNotFoundError:
                continue
            true_model_val = get_param_func(true_model)
            try:
                result = get_result(seed, n_bcode)
            except FileNotFoundError:
                continue
            fitted_val = get_param_func(result)
            dist = np.linalg.norm(fitted_val - true_model_val)
            n_bcode_results[key][idx].append(dist)

for seed in seeds:
    for idx, n_bcode in enumerate(num_barcodes):
        try:
            true_model = get_true_model(seed, n_bcode)
        except FileNotFoundError:
            continue
        true_mrca_meas = MRCADistanceMeasurer(true_model[tree_idx])
        true_bhv_meas = BHVDistanceMeasurer(true_model[tree_idx], "_output/scratch")

        plot_simulation_common.plot_mrca_matrix(
            true_mrca_meas.ref_tree_mrca_matrix,
            OUT_TRUE_MRCA_PLOT % (prefix, model_seed, seed))

        n_bcode_results["leaves"][idx].append(len(true_model[tree_idx]))

        try:
            rand_res = get_rand_tree(seed, n_bcode)
            rand_tree = rand_res[tree_idx]
        except FileNotFoundError:
            print("asdfasd")
            continue

        rand_dists = []
        rand_bhv_dists = []
        for _ in range(30):
            br_scale = 0.8
            has_neg = True
            while has_neg:
                has_neg = False
                for node in rand_tree.traverse():
                    if node.is_root():
                        continue
                    all_same_leaves = len(node.get_children()) > 0 and all([c.is_leaf() for c in node.get_children()]) and node.abundance > 1
                    if all_same_leaves or node.is_leaf():
                        node.dist = 1 - node.up.get_distance(rand_tree)
                        if node.dist < 0:
                            has_neg = True
                            break
                    else:
                        node.dist = np.random.rand() * br_scale
                br_scale *= 0.8
            dist = true_mrca_meas.get_dist(rand_tree)
            rand_dists.append(dist)
            dist = true_bhv_meas.get_dist(rand_tree)
            rand_bhv_dists.append(dist)
        n_bcode_results["random_bhv"][idx].append(np.mean(rand_bhv_dists))
        n_bcode_results["random_mrca"][idx].append(np.mean(rand_dists))

        plot_simulation_common.plot_mrca_matrix(
            true_mrca_meas._get_mrca_matrix(rand_tree),
            OUT_RAND_MRCA_PLOT % (prefix, model_seed, seed, n_bcode))

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
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["zero_bhv"][idx].append(dist)

        for node in zero_tree:
            node.dist = 0
        dist = true_bhv_meas.get_dist(zero_tree)
        n_bcode_results["super_zero_bhv"][idx].append(dist)

        try:
            result = get_result(seed, n_bcode)
        except FileNotFoundError:
            continue

        plot_simulation_common.plot_mrca_matrix(
            true_mrca_meas._get_mrca_matrix(result[tree_idx]),
            OUT_FITTED_MRCA_PLOT % (prefix, model_seed, seed, n_bcode))

        n_bcode_results["seeds"][idx].append(len(true_model[tree_idx]))

        dist = true_mrca_meas.get_dist(result[tree_idx])
        n_bcode_results["mrca"][idx].append(dist)

        dist = true_bhv_meas.get_dist(result[tree_idx])
        n_bcode_results["bhv"][idx].append(dist)

plot_simulation_common.print_results(
        num_barcodes,
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
