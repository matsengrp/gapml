import os
import json
import six
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common
from common import assign_rand_tree_lengths

np.random.seed(0)

model_seed = 7
seeds = [7]
num_barcodes = [1]
prefix = ""
growth_stage = "dome"
tree_idx = 1
num_rands = 20

TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/%s/num_barcodes%d/tune_fitted_tune_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/%s/true_model.pkl"
RAND_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/%s/num_barcodes%d/parsimony_tree0.pkl"

def get_true_model(seed, n_bcodes):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed, growth_stage)
    return plot_simulation_common.get_true_model(file_name, None, n_bcodes)

def get_result(seed, n_bcodes):
    res_file = TEMPLATE % (prefix, model_seed, seed, growth_stage, n_bcodes)
    trees = []
    scores = []
    model_params = []
    hists = []
    print(res_file)
    with open(res_file, "rb") as f:
        results = six.moves.cPickle.load(f)

        for result in results["chad_results"]:
            scores.append(result.score)
            print("bif", len(result.fit_res.fitted_bifurc_tree))
            leaved_bifurc_tree = plot_simulation_common._get_leaved_result(
                    result.fit_res.fitted_bifurc_tree)
            print("leaved", len(leaved_bifurc_tree))
            trees.append(leaved_bifurc_tree)
            model_params.append(result.fit_res.model_params_dict)
            hists.append(result.fit_res.train_history)
    return trees, scores, model_params, hists, results["chad_results"]

def get_rand_tree(seed, n_bcodes):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, growth_stage, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)

n_bcode_results = {
        "targ": [],
        "score": [],
        "internal_corr": [],
        "random_internal_corr": [],
        "bhv": [],
        "random_bhv": []}
for seed_idx, seed in enumerate(seeds):
    for idx, n_bcode in enumerate(num_barcodes):
        try:
            true_model = get_true_model(seed, n_bcode)
            tot_height = true_model[0]["tot_time"]
        except FileNotFoundError as e:
            print(e)
            continue

        try:
            trees, scores, model_params, hists, chad_results = get_result(seed, n_bcode)
        except FileNotFoundError as e:
            print(e)
            continue
        true_internal_meas = InternalCorrMeasurer(true_model[tree_idx], "_output/scratch")
        true_bhv_meas = BHVDistanceMeasurer(true_model[tree_idx], "_output/scratch")

        try:
            rand_tree = get_rand_tree(seed, n_bcode)
        except FileNotFoundError as e:
            print(e)
            continue

        rand_internal_dists = []
        rand_bhv_dists = []
        for _ in range(num_rands):
            assign_rand_tree_lengths(rand_tree[tree_idx], tot_height)

            dist = true_bhv_meas.get_dist(rand_tree[tree_idx])
            rand_bhv_dists.append(dist)
            dist = true_internal_meas.get_dist(rand_tree[tree_idx])
            rand_internal_dists.append(dist)
        n_bcode_results["random_bhv"].append(np.mean(rand_bhv_dists))
        n_bcode_results["random_internal_corr"].append(np.mean(rand_internal_dists))

        for tree, score, model_param, hist in zip(trees, scores, model_params, hists):
            dist = true_internal_meas.get_dist(tree)
            n_bcode_results["internal_corr"].append(dist)

            dist = true_bhv_meas.get_dist(tree)
            n_bcode_results["bhv"].append(dist)

            n_bcode_results["score"].append(score)

            fitted_p = plot_simulation_common.get_target_lams((model_param, None))
            true_p = plot_simulation_common.get_target_lams(true_model)
            print(fitted_p)
            n_bcode_results["targ"].append(np.linalg.norm(fitted_p - true_p))

            print("start", hist[1])
            print("end", hist[-1])

for res in chad_results:
    print(res)

print(n_bcode_results["random_bhv"])
print(n_bcode_results["random_internal_corr"])
print("corr = ", n_bcode_results["internal_corr"])
print("bhv = ", n_bcode_results["bhv"])
print("score = ", n_bcode_results["score"])
print("targ_diff = ", n_bcode_results["targ"])
