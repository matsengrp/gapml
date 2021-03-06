import numpy as np
import six
import pickle
import logging
import os
import pathlib

import itertools
from typing import List, Tuple, Dict

from constants import COLORS

def parse_comma_str(comma_str, type_cast=float):
    return [type_cast(a) for a in comma_str.split(",")]

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
    (This is also the logit function)
    @return x for prob = 1/(1 + exp(-x))
    """
    return -np.log(np.divide(1.0, prob) - 1.0)

def save_data(data, out_file: str):
    with open(out_file, "wb") as f:
        six.moves.cPickle.dump(data, f, protocol = 2)

def get_randint():
    return np.random.randint(low=0, high=10000000)

def create_directory(file_name):
    dir_path = os.path.dirname(file_name)
    print(dir_path)
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True)
            print("making directory")
    except FileExistsError:
        print("directory already exists")


"""
Random initialization code
"""
def get_init_target_lams(size, mean_val=0):
    random_perturb = np.random.uniform(size=size) * 0.001
    random_perturb = random_perturb - np.mean(random_perturb)
    random_perturb[0] = 0
    return np.exp(mean_val * np.ones(size) + random_perturb)

def assign_rand_tree_lengths(rand_tree, tot_height):
    """
    Assign random branch lengths to the fixed rand_tree topology.
    Branch lengths assigned by using tau-space parameterization
    """
    rand_tree_nodes = [node for node in rand_tree.get_descendants("preorder") if not node.is_leaf()]
    num_internal_nodes = len(rand_tree_nodes)

    internal_node_dists = np.sort(np.random.rand(num_internal_nodes)) * tot_height
    rand_tree.dist = 0
    rand_tree.add_feature("dist_to_root", 0)
    for i, node in enumerate(rand_tree_nodes):
        node.add_feature("dist_to_root", internal_node_dists[i])
        node.dist = node.dist_to_root - node.up.dist_to_root

    for leaf in rand_tree:
        leaf.dist = tot_height - leaf.up.dist_to_root

    for node in rand_tree.get_descendants():
        assert node.dist >= 0
