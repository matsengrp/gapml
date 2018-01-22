import copy
from scipy.stats import expon
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from cell_state_simulator import CellStateSimulator
from allele import Allele
from allele_events import AlleleEvents
from allele_simulator import AlleleSimulator


class CLTSimulator:
    """
    Class for simulating cell lineage trees.
    Subclass this to play around with different generative models.

    This class is generates CLT based on cell division/death/cell-type-differentiation. Allele is independently modified along branches.
    """

    def __init__(self,
        birth_rate: float,
        death_rate: float,
        cell_state_simulator: CellStateSimulator,
        allele_simulator: AlleleSimulator):
        """
        @param birth_rate: the CTMC rate param for cell division
        @param death_rate: the CTMC rate param for cell death
        @param cell_type_tree: the tree that specifies how cells differentiate
        @param allele_simulator: a simulator for how alleles get modified
        """
        self.birth_scale = 1.0 / birth_rate
        self.death_scale = 1.0 / death_rate
        self.cell_state_simulator = cell_state_simulator
        self.allele_simulator = allele_simulator

    def simulate(self, root_allele: Allele, root_cell_state: CellState, time: float):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        """
        tree = CellLineageTree(
            allele=root_allele,
            cell_state=root_cell_state,
            dist=0)

        self._simulate_tree(tree, time)

        # Need to label the leaves (alive cells only) so that we
        # can later prune the tree. Leaf names must be unique!
        for idx, leaf in enumerate(tree, 1):
            if not leaf.dead:
                leaf.name = "leaf%d" % idx

        return tree

    def _run_race(self):
        """
        Run the race to determine branch length and event at end of branch
        Does not take into account the maximum observation time!
        @return race_winner: 0 means birth, 1 means death
                branch_length: time til the next event
        """
        t_birth = expon.rvs(scale=self.birth_scale)
        t_death = expon.rvs(scale=self.death_scale)
        race_winner = np.argmin([t_birth, t_death])
        branch_length = np.min([t_birth, t_death])
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
        race_winner, branch_length = self._run_race()
        obs_branch_length = min(branch_length, time)
        remain_time = time - obs_branch_length

        branch_end_cell_state = self.cell_state_simulator.simulate(
            tree.cell_state,
            time=obs_branch_length)

        branch_end_allele = self.allele_simulator.simulate(
            tree.allele,
            time=obs_branch_length)
        # Cells are not allowed to reach the end of a branch but have allele
        # cut up into multiple pieces
        assert(len(branch_end_allele.needs_repair) == 0)

        if remain_time <= 0:
            assert remain_time == 0
            # Time of event is past observation time
            self._process_observe_end(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele)
        elif race_winner == 0:
            # Cell division
            self._process_cell_birth(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele,
                remain_time)
        else:
            # Cell died
            self._process_cell_death(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele)

    def _process_observe_end(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_cell_state: CellState,
        branch_end_allele: Allele):
        """
        Observation time is up. Stop observing cell.
        """
        child1 = CellLineageTree(
            allele=branch_end_allele,
            cell_state=branch_end_cell_state,
            dist=branch_length)
        tree.add_child(child1)

    def _process_cell_birth(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_cell_state: CellState,
        branch_end_allele: Allele,
        remain_time: float):
        """
        Cell division

        Make two cells with identical alleles and cell states
        """
        child0 = CellLineageTree(
            allele=branch_end_allele,
            cell_state=branch_end_cell_state,
            dist=branch_length)
        child1 = CellLineageTree(
            allele=branch_end_allele,
            cell_state=branch_end_cell_state,
            dist=0)
        branch_end_allele2 = copy.deepcopy(branch_end_allele)
        child2 = CellLineageTree(
            allele=branch_end_allele2,
            cell_state=branch_end_cell_state,
            dist=0)
        child0.add_child(child1)
        child0.add_child(child2)
        tree.add_child(child0)
        self._simulate_tree(child1, remain_time)
        self._simulate_tree(child2, remain_time)
        child1.delete()
        child2.delete()


    def _process_cell_death(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_cell_state: CellState,
        branch_end_allele: Allele):
        """
        Cell has died. Add a dead child cell to `tree`.
        """
        child1 = CellLineageTree(
            allele=branch_end_allele,
            cell_state=branch_end_cell_state,
            dist=branch_length,
            dead=True)
        tree.add_child(child1)
