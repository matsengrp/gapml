import os
import json
import six
import numpy as np

from tree_distance import *
from cell_lineage_tree import CellLineageTree
import plot_simulation_common

np.random.seed(0)

model_seed = 510
seeds = range(501,503)
num_barcodes = [1,3,9,27]
prefix = ""
tree_idx = 1
do_plots = False

TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_fitted_refitnew_tree0.pkl"
RAND_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/parsimony_tree0.pkl"
TRUE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/true_model.pkl"
COLL_TREE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/collapsed_tree.pkl"
OUT_TRUE_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/true_mrca.png"
OUT_FITTED_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_fitted_mrca.png"
OUT_RAND_MRCA_PLOT = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_rand_mrca.png"

def get_true_model(seed, n_bcodes, _):
    file_name = TRUE_TEMPLATE % (prefix, model_seed, seed)
    tree_file_name = COLL_TREE_TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    return plot_simulation_common.get_true_model(file_name, tree_file_name, n_bcodes)

def get_result(seed, n_bcodes, _):
    res_file = TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    return plot_simulation_common.get_result(res_file)

def get_rand_tree(seed, n_bcodes, _):
    res_file = RAND_TEMPLATE % (prefix, model_seed, seed, n_bcodes)
    return plot_simulation_common.get_rand_tree(res_file)


plot_simulation_common.gather_results(
        get_true_model,
        get_result,
        get_rand_tree,
        seeds,
        num_barcodes,
        n_bcode = None,
        tree_idx = tree_idx,
        do_plots = do_plots,
        print_keys = [
            "bhv",
            "random_bhv",
            "zero_bhv",
            "super_zero_bhv",
            "mrca",
            "zero_mrca",
            "random_mrca",
            "targ",
            #"double"
            ])
