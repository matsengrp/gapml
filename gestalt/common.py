import numpy as np
import six
import pickle
import logging

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

def save_data(data, out_file: str):
    with open(out_file, "wb") as f:
        six.moves.cPickle.dump(data, f, protocol = 2)

def get_randint():
    return np.random.randint(low=0, high=10000)

def create_directory(file_name):
    dir_path = os.path.dirname(file_name)
    print(dir_path)
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True)
            print("making directory")
    except FileExistsError:
        print("directory already exists")
