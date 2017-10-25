import numpy as np

from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from cell_state import CellTypeTree, CellType

from clt_observer import CLTObserver
from clt_estimator import CLTParsimonyEstimator

from constants import *

np.random.seed(0)

cell_type_tree = CellTypeTree(cell_type=None, rate=0.1, probability=1.0)
cell_type_tree.add_child(
    CellTypeTree(cell_type=CellType.BRAIN, rate=0, probability=0.5))
cell_type_tree.add_child(
    CellTypeTree(cell_type=CellType.EYE, rate=0, probability=0.5))

bcode_simulator = BarcodeSimulator(
    np.array([0.1] * NUM_BARCODE_V7_TARGETS), np.array([0.1, 0.1]), 0.8, 3, 3, 1)
simulator = CLTSimulator(0.5, 0.01, cell_type_tree, bcode_simulator)
clt = simulator.simulate(10)
print(clt)
obs = CLTObserver(0.5)
obs_leaves, pruned_clt = obs.observe_leaves(clt)

par_estimator = CLTParsimonyEstimator()
par_est_trees = par_est.estimate(obs_leaves)
print(pruned_clt)
print("ESTIMATES")
for par_est_t for par_est_trees:
    print(par_est_t)
