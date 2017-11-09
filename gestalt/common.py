import numpy as np
import itertools

from constants import COLORS

def get_color(cell_type):
    if cell_type is None:
        return "lightgray"
    return COLORS[cell_type - 1]

def product_list(iterables, repeat):
    return [list(b) for b in itertools.product(iterables, repeat=repeat)]
