import os
import json
import six
import numpy as np
from tree_distance import MRCADistanceMeasurer

TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale0/double10/trim_zero_pr1_1/trim_poiss3_3/trim_long_pr0_0/insert_zero_pr0/insert_poiss1/num_barcodes%d/sum_states2000/tune_fitted.pkl"
TRUE_TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale0/double10/trim_zero_pr1_1/trim_poiss3_3/trim_long_pr0_0/insert_zero_pr0/insert_poiss1/true_model.pkl"
COLL_TREE_TEMPLATE = "simulation_topol_consist/_output/%d/%s/lam_scale0/double10/trim_zero_pr1_1/trim_poiss3_3/trim_long_pr0_0/insert_zero_pr0/insert_poiss1/num_barcodes%d/collapsed_tree.pkl"

def get_true_model(seed, lambda_type, n_bcodes):
    file_name = TRUE_TEMPLATE % (seed, lambda_type)
    with open(file_name, "rb") as f:
        true_model = six.moves.cPickle.load(f)
    tree_file_name = COLL_TREE_TEMPLATE % (seed, lambda_type, n_bcodes)
    with open(tree_file_name, "rb") as f:
        true_coll_tree = six.moves.cPickle.load(f)
    return (true_model["true_model_params"], true_coll_tree)

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (seed, lambda_type, n_bcodes)
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)
    return (result.model_params_dict, result.fitted_bifurc_tree)

seeds = range(400,401)
num_barcodes = [5, 10]
#lambda_type = "random"
lambda_type = "sorted"

n_bcode_results = [[] for _ in num_barcodes]

for seed in seeds:
    for idx, n_bcode in enumerate(num_barcodes):
        try:
            true_model = get_true_model(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        true_model_meas = MRCADistanceMeasurer(true_model[1])
        print(true_model_meas.ref_tree_mrca_matrix.shape)
        try:
            result = get_result(seed, lambda_type, n_bcode)
        except FileNotFoundError:
            continue
        dist = true_model_meas.get_dist(result[1])
        n_bcode_results[idx].append(dist)

for idx, all_res in enumerate(n_bcode_results):
    print(num_barcodes[idx], np.mean(all_res), np.var(all_res), len(all_res))
