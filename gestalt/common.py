import numpy as np
from scipy.misc import factorial

import itertools
from functools import reduce
from typing import List, Tuple, Dict

from indel_sets import TargetTract

from constants import COLORS

def get_color(cell_type):
    if cell_type is None:
        return "lightgray"
    return COLORS[cell_type - 1]

def product_list(iterables, repeat):
    return [list(b) for b in itertools.product(iterables, repeat=repeat)]

def merge_target_tract_groups(tt_groups: List[Tuple[TargetTract]]):
    return reduce(lambda x,y: x + y, tt_groups, ())

def get_bounded_poisson_unstd(val: int, min_val: int, max_val: int, poisson_param: float):
    """
    @return un-normalized prob of observing `val` for a bounded poisson
    """
    if val > max_val:
        return 0
    else:
        return float(np.power(poisson_param, val - min_val))/factorial(val - min_val)

def get_bounded_poisson_prob(val: int, min_val: int, max_val: int, poisson_param: float):
    """
    @return probability of observing `val` for a poisson with min_val and max_val bounds
            and the given param
    """
    if val > max_val:
        return 0
    else:
        # TODO: do not compute this from scratch everytime!
        normalization = sum([
            get_bounded_poisson_unstd(i, min_val, max_val, poisson_param) for i in range(min_val, max_val + 1)])
        return get_bounded_poisson_unstd(val, min_val, max_val, poisson_param)/normalization
