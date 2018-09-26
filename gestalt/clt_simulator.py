import copy
from scipy.stats import expon
import numpy as np
import logging

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
    def __init__(self,
        cell_state_simulator: CellStateSimulator,
        allele_simulator: AlleleSimulator):
        """
        @param cell_type_tree: the tree that specifies how cells differentiate
        @param allele_simulator: a simulator for how alleles get modified
        """
        self.cell_state_simulator = cell_state_simulator
        self.allele_simulator = allele_simulator

    def simulate(self, root_allele: Allele, root_cell_state: CellState, tot_time: float, max_nodes: int = 50):
        raise NotImplementedError()

    def _simulate_alleles(self, tree: CellLineageTree):
        """
        assumes the tree has been built and we just simulate alleles along the branches in `tree`
        """
        for bcode_idx in range(tree.allele_list.bcode_meta.num_barcodes):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    self._simulate_branch_allele(node, bcode_idx)

    def _simulate_branch_allele(self, node: CellLineageTree, bcode_idx: int):
        """
        Simulate allele for the branch with end node `node`
        @param bcode_idx: the index of the barcode we are simulating for
        """
        allele = node.up.allele_list.alleles[bcode_idx]
        branch_end_allele = self.allele_simulator.simulate(
            allele,
            time=node.dist)
        new_allele_list = [a.allele for a in node.allele_list.alleles]
        new_allele_list[bcode_idx] = branch_end_allele.allele
        branch_end_allele_list = AlleleList(
                new_allele_list,
                allele.bcode_meta)

        node.set_allele_list(branch_end_allele_list)

    def _simulate_cell_states(self, tree: CellLineageTree):
        """
        assumes the tree has been built and we just simulate cell states along the branches in `tree`
        """
        for node in tree.traverse('preorder'):
            if not node.is_root():
                branch_end_cell_state = self.cell_state_simulator.simulate(
                    node.up.cell_state,
                    time=node.dist)
                node.cell_state = branch_end_cell_state

class BirthDeathTreeSimulator:
    """
    Class for creating cell lineage trees without allele or cell types
    """
    def __init__(self,
            birth_sync_rounds: int,
            birth_sync_time: float,
            birth_decay: float,
            birth_min: float,
            death_rate: float):
        """
        @param birth_sync_rounds: number of synchronous cell divisions (before async begins)
        @param birth_sync_time: time between synchronous cell divisions
        @param start_birth_rate: the starting CTMC rate param for cell division
        @param birth_decay: the exponential decay rate for cell division
        @param death_rate: the CTMC rate param for cell death
        """
        self.birth_sync_rounds = birth_sync_rounds
        self.birth_sync_time = birth_sync_time
        # First round is not counted
        self.birth_async_start_time = birth_sync_time * birth_sync_rounds + 1e-10
        self.start_birth_rate = 1.0/birth_sync_time
        self.birth_decay = birth_decay
        self.birth_min = birth_min
        self.death_scale = 1.0 / death_rate

    def simulate(self, root_allele_list: AlleleList, time: float, max_nodes: int = 10):
        """
        Generates a CLT based on the model
        The root_allele and root_cell_state is constant (used as a dummy only).

        @param root_allele_list: this is copied as the AlleleList for all nodes in this tree
        @param time: amount of time to simulate the CLT
        @param max_nodes: maximum number of nodes to have in the full CLT

        @return CellLineageTree from the birth death tree simulator -- provides a topology only
        """
        self.tot_time = time
        tree = CellLineageTree(
            allele_list=root_allele_list,
            dist=0)

        self.curr_nodes = 1
        self.max_nodes = max_nodes
        self._simulate_tree(tree, time)
        return tree

    def _run_race(self, birth_scale, death_scale):
        """
        Run the race to determine branch length and event at end of branch
        Does not take into account the maximum observation time!
        @return race_winner: True means cell division happens, False means cell doesn't (hence dies)
                branch_length: time til the next event
        """
        # Birth rate very high initially?
        t_birth = expon.rvs(scale=birth_scale)
        t_death = expon.rvs(scale=death_scale)
        division_happens = t_birth < t_death
        branch_length = np.min([t_birth, t_death])
        return division_happens, branch_length

    def _simulate_tree(self, tree: CellLineageTree, remain_time: float):
        """
        The recursive function that actually makes the tree

        @param tree: the root node to create a tree from
        @param time: the max amount of time to simulate from this node
        """
        self.curr_nodes += 1
        if self.curr_nodes > self.max_nodes:
            raise ValueError("too many nodes %d", self.curr_nodes)
            return

        if remain_time == 0:
            print("CLT Simulator time out")
            return

        # Determine branch length and event at end of branch
        if self.tot_time - remain_time < self.birth_async_start_time:
            division_happens = True
            branch_length = self.birth_sync_time
        else:
            time_since_decay_begin = (self.tot_time - remain_time) - self.birth_async_start_time
            curr_birth_scale = 1.0/max(
                self.birth_min,
                self.start_birth_rate * np.exp(self.birth_decay * time_since_decay_begin))
            division_happens, branch_length = self._run_race(curr_birth_scale, self.death_scale)
        obs_branch_length = min(branch_length, remain_time)
        remain_time = remain_time - obs_branch_length

        if remain_time <= 0:
            assert remain_time == 0
            # Time of event is past observation time
            self._process_observe_end(
                tree,
                obs_branch_length,
                tree.allele_list)
        elif division_happens:
            # Cell division
            self._process_cell_birth(
                tree,
                obs_branch_length,
                tree.allele_list,
                remain_time)
        else:
            # Cell died
            self._process_cell_death(
                tree,
                obs_branch_length,
                tree.allele_list)

    def _process_observe_end(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_allele_list: AlleleList):
        """
        Observation time is up. Stop observing cell.
        """
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
            dist=branch_length)
        tree.add_child(child1)

    def _process_cell_birth(
        self,
        tree: CellLineageTree,
        branch_length: float,
        branch_end_allele_list: AlleleList,
        remain_time: float):
        """
        Cell division

        Make two cells with identical allele_lists and cell states
        """
        child0 = CellLineageTree(
            allele_list=branch_end_allele_list,
            dist=branch_length)
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
            dist=0)
        branch_end_allele_list2 = copy.deepcopy(branch_end_allele_list)
        child2 = CellLineageTree(
            allele_list=branch_end_allele_list2,
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
        branch_end_allele_list: AlleleList):
        """
        Cell has died. Add a dead child cell to `tree`.
        """
        child1 = CellLineageTree(
            allele_list=branch_end_allele_list,
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
            birth_sync_rounds: int,
            birth_sync_time: float,
            birth_decay: float,
            birth_min: float,
            death_rate: float,
            cell_state_simulator: CellStateSimulator,
            allele_simulator: AlleleSimulator):
        """
        @param birth_rate: the CTMC rate param for cell division
        @param death_rate: the CTMC rate param for cell death
        @param cell_type_tree: the tree that specifies how cells differentiate
        @param allele_simulator: a simulator for how alleles get modified
        """
        self.birth_sync_rounds = birth_sync_rounds
        self.birth_sync_time = birth_sync_time
        self.birth_decay = birth_decay
        self.birth_min = birth_min
        self.death_rate = death_rate
        self.cell_state_simulator = cell_state_simulator
        self.allele_simulator = allele_simulator

    def simulate(self,
            tree_seed: int,
            data_seed: int,
            tot_time: float,
            max_nodes: int = 10,
            max_tries: int = 3,
            dominating_percent: float = 0.8):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        @param max_nodes: maximum number of CLT nodes to simulate
        @param max_tries: number of times we try to resimulate the alleles to satisfy
                        our allele event "diversity" requirement
        @param dominating_percent: if an event appears in at least this percentage of the observed leaves,
                                then we are going to resimulate the alleles along the tree.
        """
        np.random.seed(tree_seed)
        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()
        # Run the simulation to just create the tree topology
        bd_tree_simulator = BirthDeathTreeSimulator(
                self.birth_sync_rounds,
                self.birth_sync_time,
                self.birth_decay,
                self.birth_min,
                self.death_rate)
        tree = bd_tree_simulator.simulate(
                root_allele,
                tot_time,
                max_nodes)
        tree.cell_state = root_cell_state
        num_leaves = len(tree)
        print("TOTAL tree leaves", len(tree))

        np.random.seed(data_seed)
        for _ in range(max_tries):
            # Run the simulation to create the alleles along the tree topology
            self._simulate_alleles(tree)
            # Check there is no event super early on that everyone shares this same evt
            # Otherwise learning will be really hard. This simulated tree
            # also won't look anything like real data.
            intersection_evt_count = {}
            for leaf in tree:
                for evt in leaf.allele_events_list[0].events:
                    if evt in intersection_evt_count:
                        intersection_evt_count[evt] += 1
                    else:
                        intersection_evt_count[evt] = 1
            no_dominating_evt = True
            for evt, val in intersection_evt_count.items():
                if val > num_leaves * dominating_percent:
                    no_dominating_evt = False
                    break

            if no_dominating_evt:
                self._simulate_cell_states(tree)
                break
        return tree
