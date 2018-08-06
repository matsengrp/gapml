import os
import json
import six
import numpy as np
from tree_distance import MRCADistanceMeasurer, RootRFDistanceMeasurer

#lambda_type = "random"
lambda_type = "sorted"
lam_scale = 0
double = 10
trim_zero_pr = 1
trim_poiss = 3
trim_long_pr = 0
insert_zero_pr = 0
insert_poiss = 1

TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale%d/double%d/trim_zero_pr%d_%d/trim_poiss%d_%d/trim_long_pr%d_%d/insert_zero_pr%d/insert_poiss%d/num_barcodes%d/sum_states2000/tune_fitted.pkl"
TRUE_TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale%d/double%d/trim_zero_pr%d_%d/trim_poiss%d_%d/trim_long_pr%d_%d/insert_zero_pr%d/insert_poiss%d/true_model.pkl"
COLL_TREE_TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale%d/double%d/trim_zero_pr%d_%d/trim_poiss%d_%d/trim_long_pr%d_%d/insert_zero_pr%d/insert_poiss%d/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (seed, lambda_type, lam_scale, double, trim_zero_pr, trim_zero_pr, trim_poiss, trim_poiss, trim_long_pr, trim_long_pr, insert_zero_pr, insert_poiss)
    with open(file_name, "rb") as f:
        true_model = six.moves.cPickle.load(f)
    tree_file_name = COLL_TREE_TEMPLATE % (seed, lambda_type, lam_scale, double, trim_zero_pr, trim_zero_pr, trim_poiss, trim_poiss, trim_long_pr, trim_long_pr, insert_zero_pr, insert_poiss, n_bcodes)
    with open(tree_file_name, "rb") as f:
        true_coll_tree = six.moves.cPickle.load(f)
    return (true_model["true_model_params"], true_coll_tree)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (seed, lambda_type, lam_scale, double, trim_zero_pr, trim_zero_pr, trim_poiss, trim_poiss, trim_long_pr, trim_long_pr, insert_zero_pr, insert_poiss, n_bcodes)
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)
    return (result.model_params_dict, result.fitted_bifurc_tree)

def get_target_lams(model_param_tuple):
    return model_param_tuple[0]["target_lams"]

def get_double_cut_weight(model_param_tuple):
    return model_param_tuple[0]["double_cut_weight"]

seeds = range(400,410)
num_barcodes = [5, 10, 20, 40]

get_param_func_dict = {
        "mrca": None, # custom function
        "rf": None, # custom function
        "targ": get_target_lams,
        "double": get_double_cut_weight}

n_bcode_results = {
        key: [[] for _ in num_barcodes]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    if get_param_func is None:
        continue

    for seed in seeds:
        for idx, n_bcode in enumerate(num_barcodes):
            true_model = get_true_model(seed, lambda_type, n_bcode)
            true_model_val = get_param_func(true_model)
            try:
                result = get_result(seed, lambda_type, n_bcode)
            except Exception:
                # file not there
                continue
            fitted_val = get_param_func(result)
            dist = np.linalg.norm(fitted_val - true_model_val)
            n_bcode_results[key][idx].append(dist)

for seed in seeds:
    for idx, n_bcode in enumerate(num_barcodes):
        true_model = get_true_model(seed, lambda_type, n_bcode)
        true_mrca_meas = MRCADistanceMeasurer(true_model[1])
        print(true_mrca_meas.ref_tree_mrca_matrix.shape)
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except Exception:
            # file not there
            continue
        dist = true_mrca_meas.get_dist(result[1])
        n_bcode_results["mrca"][idx].append(dist)

        true_rf_meas = RootRFDistanceMeasurer(true_model[1], None)
        dist = true_rf_meas.get_dist(result[1])
        n_bcode_results["rf"][idx].append(dist)

for idx, n_bcode in enumerate(num_barcodes):
    size = len(n_bcode_results["mrca"][idx])
    print("%s & %d & %d & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f)" % (
        lambda_type,
        n_bcode,
        size,
        np.mean(n_bcode_results["mrca"][idx]),
        np.sqrt(np.var(n_bcode_results["mrca"][idx])/size),
        np.mean(n_bcode_results["rf"][idx]),
        np.sqrt(np.var(n_bcode_results["rf"][idx])/size),
        np.mean(n_bcode_results["targ"][idx]),
        np.sqrt(np.var(n_bcode_results["targ"][idx])/size),
        np.mean(n_bcode_results["double"][idx]),
        np.sqrt(np.var(n_bcode_results["double"][idx])/size),
    ))
