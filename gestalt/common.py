import numpy as np
import pickle

import itertools
from typing import List, Tuple, Dict

from constants import COLORS

def get_color(cell_type):
    if cell_type is None:
        return "lightgray"
    return COLORS[cell_type - 1]

def product_list(iterables, repeat):
    return [list(b) for b in itertools.product(iterables, repeat=repeat)]

def sigmoid(x: float):
    return 1.0/(1.0 + np.exp(-x))

def inv_sigmoid(prob: float):
    """
    @return x for prob = 1/(1 + exp(-x))
    """
    return -np.log(np.divide(1.0, prob) - 1.0)

def save_model_data(file_name, model_vars, cell_type_tree, obs_leaves, true_tree, clt):
    with open(file_name, "wb") as f:
        pickle.dump({
            "model_vars": model_vars,
            "cell_type_tree": cell_type_tree,
            "obs_leaves": obs_leaves,
            "true_tree": true_tree,
            "clt": clt,
        }, f, protocol=-1)

def save_fitted_models(
        file_name,
        fitted_models_dict):
    """
    @param file_name: str, file to save to
    @param fitted_models_dict: dictionary mapping rf distance to a list of
                tuples. Each tuple has its first elem as the penalized log lik score
                and the second elem as the CLTLikelihoodEstimator

    Pickles the models (while avoiding tensorflow unhappiness)
    """
    res_dict = {}
    for k, v in fitted_models_dict.items():
        res_dict[k] = [
                (model_res[0], model_res[1].get_vars_as_dict())
                for model_res in v]

    with open(file_name, "wb") as f:
        pickle.dump(res_dict, f, protocol=-1)

def get_rf_dist_dict(trees, true_tree):
    # Now calculate the rf distances of each random tree
    rf_tree_dict = {}
    for tree in trees:
        rf_res = true_tree.robinson_foulds(
                tree,
                attr_t1="allele_events_list_str",
                attr_t2="allele_events_list_str",
                expand_polytomies=False,
                unrooted_trees=False)
        print("rf dist", rf_res[0])
        rf_dist = rf_res[0]
        if rf_dist in rf_tree_dict:
            rf_tree_dict[rf_dist].append(tree)
        else:
            rf_tree_dict[rf_dist] = [tree]

    return rf_tree_dict
