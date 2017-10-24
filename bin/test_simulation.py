import numpy as np

from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from cell_state import CellTypeTree, CellType

from constants import *

cell_type_tree = CellTypeTree(cell_type=None, rate=0.1, probability=1.0)
cell_type_tree.add_child(
    CellTypeTree(cell_type=CellType.BRAIN, rate=0, probability=0.5))
cell_type_tree.add_child(
    CellTypeTree(cell_type=CellType.EYE, rate=0, probability=0.5))
print(cell_type_tree)

bcode_simulator = BarcodeSimulator(
    np.array([0.1] * NUM_BARCODE_V7_TARGETS), np.array([0.1, 0.1]), 1, 3, 1)
simulator = CLTSimulator(0.5, 0.01, cell_type_tree, bcode_simulator)
clt = simulator.simulate(4)
print(clt)
