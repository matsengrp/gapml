import os
import json
import six
import numpy as np

from tree_distance import MRCADistanceMeasurer, MRCASpearmanMeasurer, RootRFDistanceMeasurer
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

seeds = range(0,5)
n_bcode = 1
double_cuts = [2, 8, 32]

TEMPLATE = "simulation_topology_double/_output/model_seed600/%d/double_cut%d/num_barcodes%d/sum_states2000/tune_fitted.pkl"
TRUE_TEMPLATE = "simulation_topology_double/_output/model_seed600/%d/double_cut%d/true_model.pkl"
OBS_TEMPLATE = "simulation_topology_double/_output/model_seed600/%d/double_cut%d/num_barcodes%d/obs_data.pkl"
COLL_TREE_TEMPLATE = "simulation_topology_double/_output/model_seed600/%d/double_cut%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (seed, double)
    tree_file_name = COLL_TREE_TEMPLATE % (seed, lambda_type, double, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (seed, double, n_bcodes)
    return plot_simulation_common.get_result(res_file)

get_param_func_dict = {
        "mrca": None, # custom function
        "rf": None, # custom function
        "targ": plot_simulation_common.get_target_lams,
        "double": plot_simulation_common.get_double_cut_weight,
        "leaves": None}

n_bcode_results = {
        key: [[] for _ in double_cuts]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, lambda_type in enumerate(double_cuts):
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
    for idx, lambda_type in enumerate(double_cuts):
        try:
            true_model = get_true_model(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        true_mrca_meas = MRCADistanceMeasurer(true_model[1])
        n_bcode_results["leaves"][idx].append(len(true_model[2]))
        #print("true...", true_mrca_meas.ref_tree_mrca_matrix.shape)
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        dist = true_mrca_meas.get_dist(result[1])
        n_bcode_results["mrca"][idx].append(dist)

        true_rf_meas = MRCASpearmanMeasurer(true_model[1], None)
        dist = true_rf_meas.get_dist(result[1])
        #print("RF", dist)
        n_bcode_results["rf"][idx].append(dist)

for idx, double_cut in enumerate(double_cuts):
    size = len(n_bcode_results["mrca"][idx])
    print("%d & %d & %d & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f)" % (
        double_cut,
        n_bcode,
        size,
        np.mean(n_bcode_results["leaves"][idx]),
        np.sqrt(np.var(n_bcode_results["leaves"][idx])/size),
        np.mean(n_bcode_results["mrca"][idx]),
        np.sqrt(np.var(n_bcode_results["mrca"][idx])/size),
        np.mean(n_bcode_results["rf"][idx]),
        np.sqrt(np.var(n_bcode_results["rf"][idx])/size),
        np.mean(n_bcode_results["targ"][idx]),
        np.sqrt(np.var(n_bcode_results["targ"][idx])/size),
        np.mean(n_bcode_results["double"][idx]),
        np.sqrt(np.var(n_bcode_results["double"][idx])/size),
    ))
