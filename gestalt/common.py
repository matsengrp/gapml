import numpy as np

from constants import COLORS

def get_color(cell_type):
    if cell_type is None:
        return "lightgray"
    return COLORS[cell_type - 1]
