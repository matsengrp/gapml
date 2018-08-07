import os
import json
import six
import numpy as np

TEMPLATE = "simulation_consistency/_output/seed%d/%s/lambda_scale_%d/double_cut_%d/trim_zero_prob_%d_%d/trim_poissons_%d_%d/trim_long_prob_%d_%d/insert_zero_prob_%d/insert_poisson_%d/num_inits_3/num_barcodes%d/sum_states2000/tree0_fitted.pkl"
TRUE_TEMPLATE = "simulation_consistency/_output/seed%d/%s/lambda_scale_%d/double_cut_%d/trim_zero_prob_%d_%d/trim_poissons_%d_%d/trim_long_prob_%d_%d/insert_zero_prob_%d/insert_poisson_%d/num_inits_3/true_model.pkl"

def get_true_model(seed, lambda_type):
    file_name = TRUE_TEMPLATE % (seed, lambda_type, lambda_scale, double_cut, trim_zero_prob, trim_zero_prob, trim_poisson, trim_poisson, trim_long_prob, trim_long_prob, insert_zero_prob, insert_poisson)
    with open(file_name, "rb") as f:
        true_model = six.moves.cPickle.load(f)
    return (true_model["true_model_params"], true_model["true_subtree"])

def get_result(seed, lambda_type, n_bcodes):
    res_file = TEMPLATE % (seed, lambda_type, lambda_scale, double_cut, trim_zero_prob, trim_zero_prob, trim_poisson, trim_poisson, trim_long_prob, trim_long_prob, insert_zero_prob, insert_poisson, n_bcodes)
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)
    return (result["refit"].model_params_dict, result["refit"].fitted_bifurc_tree)

def get_target_lams(model_param_tuple):
    return model_param_tuple[0]["target_lams"]

def get_double_cut_weight(model_param_tuple):
    return model_param_tuple[0]["double_cut_weight"]

def get_branch_lens(model_param_tuple, node_idxs = [1]):
    br_len_dict = {i: 0 for i in node_idxs}
    for node in model_param_tuple[1].traverse():
        if node.node_id in node_idxs:
            br_len_dict[node.node_id] = node.dist

    br_len_list = []
    for i in node_idxs:
        br_len_list.append(br_len_dict[i])
    return np.array(br_len_list)

seeds = range(500,505)
num_barcodes = [2, 5, 10,20,40]
lambda_type = "sorted"
lambda_scale = 1
double_cut = 10
trim_zero_prob = 1
trim_poisson = 3
trim_long_prob = 0
insert_zero_prob = 0
insert_poisson = 1
#lambda_type = "sorted"
get_param_func_dict = {
        "br": get_branch_lens,
        "targ": get_target_lams,
        "double": get_double_cut_weight}

n_bcode_results = {
        key: [[] for _ in num_barcodes]
        for key in get_param_func_dict.keys()}

for key in get_param_func_dict.keys():
    get_param_func = get_param_func_dict[key]
    for seed in seeds:
        true_model = get_true_model(seed, lambda_type)
        true_model_val = get_param_func(true_model)
        for idx, n_bcode in enumerate(num_barcodes):
            try:
                result = get_result(seed, lambda_type, n_bcode)
            except Exception:
                # File not there
                continue
            fitted_val = get_param_func(result)
            dist = np.linalg.norm(fitted_val - true_model_val)
            n_bcode_results[key][idx].append(dist)

for idx, n_bcode in enumerate(num_barcodes):
    size = len(n_bcode_results["br"][idx])
    print("%s & %d & %d & %.04f (%.04f) & %.04f (%.04f) & %.04f (%.04f)\\\\" % (
        lambda_type,
        n_bcode,
        size,
        np.mean(n_bcode_results["br"][idx]),
        np.sqrt(np.var(n_bcode_results["br"][idx])/size),
        np.mean(n_bcode_results["targ"][idx]),
        np.sqrt(np.var(n_bcode_results["targ"][idx])/size),
        np.mean(n_bcode_results["double"][idx]),
        np.sqrt(np.var(n_bcode_results["double"][idx])/size),
    ))
