import copy
from scipy.stats import expon
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from cell_state_simulator import CellStateSimulator
from allele import Allele, AlleleList
from allele_events import AlleleEvents
from allele_simulator import AlleleSimulator

from common import sigmoid

class CLTSimulator:
    """
    Class for creating cell lineage trees with allele and cell state information
    """
    def simulate(self, root_allele: Allele, root_cell_state: CellState, time: float, max_nodes: int = 10):
        raise NotImplementedError()

class BirthDeathTreeSimulator:
    """
    Class for creating cell lineage trees without allele or cell types
    """
    def __init__(self,
        birth_rate: float,
        death_rate: float):
        """
        @param birth_rate: the CTMC rate param for cell division
        @param death_rate: the CTMC rate param for cell death
        """
        self.birth_scale = 1.0 / birth_rate
        self.death_scale = 1.0 / death_rate

    def simulate(self, root_allele_list: AlleleList, root_cell_state: CellState, time: float, max_nodes: int = 10):
        """
        Generates a CLT based on the model
        The root_allele and root_cell_state is constant (used as a dummy only).
        Override them if you want them to change

        @param root_allele: ignored
        @param root_cell_state: ignored
        @param time: amount of time to simulate the CLT
        """
        tree = CellLineageTree(
            allele_list=root_allele_list,
            cell_state=root_cell_state,
            dist=0)

        self.curr_nodes = 1
        self.max_nodes = max_nodes
        self._simulate_tree(tree, time)

        # Need to label the leaves (alive cells only) so that we
        # can later prune the tree. Leaf names must be unique!
        for idx, leaf in enumerate(tree, 1):
            if not leaf.dead:
                leaf.name = "leaf%d" % idx

        return tree

    def _run_race(self, time:float):
        """
        Run the race to determine branch length and event at end of branch
        Does not take into account the maximum observation time!
        @return race_winner: True means cell division happens, False means cell doesn't (hence dies)
                branch_length: time til the next event
        """
        # Birth rate very high initially?
        t_birth = expon.rvs(scale=self.birth_scale * sigmoid(-1 + time))
        t_death = expon.rvs(scale=self.death_scale)
        division_happens = t_birth < t_death
        branch_length = np.min([t_birth, t_death])
        return division_happens, branch_length

    def _simulate_tree(self, tree: CellLineageTree, time: float):
        """
        The recursive function that actually makes the tree

        @param tree: the root node to create a tree from
        @param time: the max amount of time to simulate from this node
        """
        self.curr_nodes += 1
        if self.curr_nodes > self.max_nodes:
            print("too many nodes")
            return

        if time == 0:
            print("time out")
            return

        # Determine branch length and event at end of branch
        division_happens, branch_length = self._run_race(time)
        obs_branch_length = min(branch_length, time)
        remain_time = time - obs_branch_length

        # Keep allele/cell state constant
        branch_end_cell_state = tree.cell_state
        branch_end_allele_list = tree.allele_list

        if remain_time <= 0:
            assert remain_time == 0
            # Time of event is past observation time
            self._process_observe_end(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele_list)
        elif division_happens:
            # Cell division
            self._process_cell_birth(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele_list,
                remain_time)
        else:
            # Cell died
            self._process_cell_death(
                tree,
                obs_branch_length,
                branch_end_cell_state,
                branch_end_allele_list)

    def _process_observe_end(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_cell_state: CellState,
        branch_end_allele_list: Allele):
        """
        Observation time is up. Stop observing cell.
        """
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
            cell_state=branch_end_cell_state,
            dist=branch_length)
        tree.add_child(child1)

    def _process_cell_birth(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_cell_state: CellState,
        branch_end_allele_list: Allele,
        remain_time: float):
        """
        Cell division

        Make two cells with identical allele_lists and cell states
        """
        child0 = CellLineageTree(
            allele_list=branch_end_allele_list,
            cell_state=branch_end_cell_state,
            dist=branch_length)
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
            cell_state=branch_end_cell_state,
            dist=0)
        branch_end_allele_list2 = copy.deepcopy(branch_end_allele_list)
        child2 = CellLineageTree(
            allele_list=branch_end_allele_list2,
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
        branch_end_allele_list: Allele):
        """
        Cell has died. Add a dead child cell to `tree`.
        """
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
            cell_state=branch_end_cell_state,
            dist=branch_length,
            dead=True)
        tree.add_child(child1)

class CLTSimulatorBifurcating(CLTSimulator, BirthDeathTreeSimulator):
    """
    Class for simulating cell lineage trees.
    Subclass this to play around with different generative models.

    This class is generates CLT based on cell division/death. Allele is independently modified along branches.
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
        self.bd_tree_simulator = BirthDeathTreeSimulator(birth_rate, death_rate)
        self.cell_state_simulator = cell_state_simulator
        self.allele_simulator = allele_simulator

    def simulate(self,
            tree_seed: int,
            data_seed: int,
            time: float,
            max_nodes: int = 10):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        """
        np.random.seed(tree_seed)
        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()
        tree = self.bd_tree_simulator.simulate(
                root_allele,
                root_cell_state,
                time,
                max_nodes)

        np.random.seed(data_seed)
        for node in tree.traverse('preorder'):
            if not node.is_root():
                self._simulate_branch(node)

        return tree

    def _simulate_branch(self, node: CellLineageTree):
        """
        The recursive function that actually makes the tree

        @param tree: the root node to create a tree from
        @param time: the max amount of time to simulate from this node
        """
        branch_end_cell_state = self.cell_state_simulator.simulate(
            node.up.cell_state,
            time=node.dist)

        branch_end_allele_list = self.allele_simulator.simulate(
            node.up.allele_list,
            time=node.dist)

        node.allele_list = branch_end_allele_list
        node.allele_events_list = branch_end_allele_list.get_event_encoding()
        node.cell_state = branch_end_cell_state
