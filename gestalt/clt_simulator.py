import copy
from scipy.stats import expon
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from barcode import Barcode
from barcode_events import BarcodeEvents
from barcode_simulator import BarcodeSimulator


class CLTSimulator:
    """
    Class for simulating cell lineage trees.
    Subclass this to play around with different generative models.

    This class is generates CLT based on cell division/death/cell-type-differentiation. Barcode is independently modified along branches.
    """

    def __init__(self,
        birth_rate: float,
        death_rate: float,
        cell_type_tree: CellTypeTree,
        bcode_simulator: BarcodeSimulator):
        """
        @param birth_rate: the CTMC rate param for cell division
        @param death_rate: the CTMC rate param for cell death
        @param cell_type_tree: the tree that specifies how cells differentiate
        @param bcode_simulator: a simulator for how barcodes get modified
        """
        self.birth_scale = 1.0 / birth_rate
        self.death_scale = 1.0 / death_rate
        self.cell_type_tree = cell_type_tree
        self.bcode_simulator = bcode_simulator

    def simulate(self, time: float):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        """
        root_barcode = Barcode()
        cell_state = CellState(categorical=self.cell_type_tree)
        tree = CellLineageTree(
            barcode=root_barcode,
            cell_state=cell_state,
            dist=0)

        self._simulate_tree(tree, time)

        # Need to label the leaves (alive cells only) so that we
        # can later prune the tree. Leaf names must be unique!
        for idx, leaf in enumerate(tree, 1):
            if not leaf.dead:
                leaf.name = "leaf%d" % idx

        return tree

    def _run_race(self, cell_state):
        """
        Run the race to determine branch length and event at end of branch
        Does not take into account the maximum observation time!
        @return race_winner: 0 means birth, 1 means death, 2 means speciate
                branch_length: time til the next event
        """
        t_birth = expon.rvs(scale=self.birth_scale)
        t_death = expon.rvs(scale=self.death_scale)
        speciate_scale = cell_state.categorical_state.scale
        t_speciate = expon.rvs(
            scale=speciate_scale) if speciate_scale is not None else np.inf
        race_winner = np.argmin([t_birth, t_death, t_speciate])
        branch_length = np.min([t_birth, t_death, t_speciate])
        return race_winner, branch_length


    def _simulate_tree(self, tree: CellLineageTree, time: float):
        """
        The recursive function that actually makes the tree

        @param tree: the root node to create a tree from
        @param time: the max amount of time to simulate from this node
        """
        if time == 0:
            return

        # Determine branch length and event at end of branch
        race_winner, branch_length = self._run_race(tree.cell_state)
        obs_branch_length = min(branch_length, time)
        remain_time = time - obs_branch_length

        branch_end_barcode = self.bcode_simulator.simulate(
            tree.barcode,
            time=obs_branch_length)
        # Cells are not allowed to reach the end of a branch but have barcode
        # cut up into multiple pieces
        assert(len(branch_end_barcode.needs_repair) == 0)

        if remain_time <= 0:
            # Time of event is past observation time
            self._process_observe_end(
                tree,
                obs_branch_length,
                branch_end_barcode)
        elif race_winner == 0:
            # Cell division
            self._process_cell_birth(
                tree,
                obs_branch_length,
                branch_end_barcode,
                remain_time)
        elif race_winner == 1:
            # Cell died
            self._process_cell_death(
                tree,
                obs_branch_length,
                branch_end_barcode)
        else:
            # Cell-type differentiation
            self._process_speciate(
                tree,
                obs_branch_length,
                branch_end_barcode,
                remain_time)

    def _process_observe_end(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_barcode: Barcode):
        """
        Observation time is up. Stop observing cell.
        """
        child1 = CellLineageTree(
            barcode=branch_end_barcode,
            cell_state=tree.cell_state,
            dist=branch_length)
        tree.add_child(child1)

    def _process_cell_birth(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_barcode: Barcode,
        remain_time: float):
        """
        Cell division

        Make two cells with identical barcodes and cell states
        """
        child1 = CellLineageTree(
            barcode=branch_end_barcode,
            cell_state=tree.cell_state,
            dist=branch_length)
        branch_end_barcode2 = copy.deepcopy(branch_end_barcode)
        child2 = CellLineageTree(
            barcode=branch_end_barcode2,
            cell_state=tree.cell_state,
            dist=branch_length)
        tree.add_child(child1)
        tree.add_child(child2)
        self._simulate_tree(child1, remain_time)
        self._simulate_tree(child2, remain_time)

    def _process_cell_death(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_barcode: Barcode):
        """
        Cell has died. Add a dead child cell to `tree`.
        """
        child1 = CellLineageTree(
            barcode=branch_end_barcode,
            cell_state=tree.cell_state,
            dist=branch_length,
            dead=True)
        tree.add_child(child1)

    def _process_speciate(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_barcode: Barcode,
        remain_time: float):
        """
        Cell differentiates
        Two children cells with same barcode, descendant cell types
            from cell type tree. Pick descendant cell type at random.
        """
        # Decide children cell types
        child_cell_types = tree.cell_state.categorical_state.children
        child_type_probs = [c.probability for c in child_cell_types]
        cell_type1, cell_type2 = np.random.choice(
            a=len(child_type_probs),
            size=len(child_cell_types),
            p=child_type_probs)

        # Create child 1
        child1 = CellLineageTree(
            barcode=branch_end_barcode,
            cell_state=CellState(categorical=child_cell_types[cell_type1]),
            dist=branch_length)
        tree.add_child(child1)

        # Create child 2 - make sure its barcode is a new object
        branch_end_barcode2 = copy.deepcopy(branch_end_barcode)
        child2 = CellLineageTree(
            barcode=branch_end_barcode2,
            cell_state=CellState(categorical=child_cell_types[cell_type2]),
            dist=branch_length)
        tree.add_child(child2)

        # Recurse
        self._simulate_tree(child1, remain_time)
        self._simulate_tree(child2, remain_time)
