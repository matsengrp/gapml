import numpy as np

import itertools
from functools import reduce
from typing import List, Tuple, Dict

from indel_sets import TargetTract, TargetTractRepr

from constants import COLORS

def get_color(cell_type):
    if cell_type is None:
        return "lightgray"
    return COLORS[cell_type - 1]

def product_list(iterables, repeat):
    return [list(b) for b in itertools.product(iterables, repeat=repeat)]

def merge_target_tract_groups(tt_groups: List[Tuple[TargetTract]]):
    """
    @return flattened version of a list of tuples of target tract
    """
    tts_raw = reduce(lambda x,y: x + y, tt_groups, ())
    return TargetTractRepr(*tts_raw)

def target_tract_repr_diff(tts1: Tuple[TargetTract], tts2: Tuple[TargetTract]):
    """
    Assumes that tts1 <= tts2
    @return tts2 - tts1
    """
    idx1 = 0
    idx2 = 0
    n1 = len(tts1)
    n2 = len(tts2)
    new_tuple = ()
    while idx1 < n1 and idx2 < n2:
        tt1 = tts1[idx1]
        tt2 = tts2[idx2]

        if tt2.max_target < tt1.min_target:
            new_tuple += (tt2,)
            idx2 += 1
            continue

        # Now we have overlapping events
        idx1 += 1
        idx2 += 1
        if tt1 != tt2:
            new_tuple += (tt2,)

    new_tuple += tts2[idx2:]

    return new_tuple

def inv_sigmoid(prob: float):
    """
    @return x for prob = 1/(1 + exp(-x))
    """
    return -np.log(np.divide(1.0, prob) - 1.0)
