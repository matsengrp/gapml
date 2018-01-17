import numpy as np
import itertools
from functools import reduce
from typing import List, Tuple

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
