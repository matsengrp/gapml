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
