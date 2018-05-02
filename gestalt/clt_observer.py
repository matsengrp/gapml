from typing import List, Dict, Tuple
import numpy as np
import logging
import random

from allele import Allele, AlleleList
from allele_events import AlleleEvents
from cell_state import CellState
from cell_lineage_tree import CellLineageTree
from alignment import Aligner
import collapsed_tree

class ObservedAlignedSeq:
    def __init__(self,
            allele_list: AlleleList,
            allele_events_list: List[AlleleEvents],
            cell_state: CellState,
            abundance: float):
        """
        Stores alleles that are observed.
        Since these are stored using the allele class,
        it implicitly gives us the alignment
        """
        self.allele_list = allele_list
        self.allele_events_list = allele_events_list
        self.cell_state = cell_state
        self.abundance = abundance

    def __str__(self):
        return "||".join([str(allele_evts) for allele_evts in self.allele_events_list])

class CLTObserver:
    def __init__(self,
                 error_rate: float = 0,
                 aligner: Aligner = None):
        """
        @param error_rate: sequencing error, introduce alternative bases uniformly at this rate
        """
        assert (0 <= error_rate <= 1)
        self.error_rate = error_rate
        self.aligner = aligner

    @staticmethod
    def _sample_leaf_ids(clt: CellLineageTree, sampling_rate: float):
        """
        Determine which leaves we observe for that sampling rate in that tree
        @return set with leaf node ids
        """
        # Stores only leaf node ids!
        observed_leaf_ids = set()
        # First sample the leaves
        for leaf in clt:
            if not leaf.dead:
                is_sampled = np.random.binomial(1, sampling_rate)
                if is_sampled:
                    observed_leaf_ids.add(leaf.node_id)
        return observed_leaf_ids
    
    @staticmethod
    def _sample_leaves(clt_orig: CellLineageTree, sampling_rate: float):
        """
        Makes a copy of the original tree, samples leaves, and returns back a new tree
        @return CellLineageTree that represents a subsampled `clt_orig`
        """
        clt = clt_orig.copy()
        observed_leaf_ids = CLTObserver._sample_leaf_ids(clt, sampling_rate)

        # Now prune the tree -- custom pruning (cause ete was doing weird things...)
        # TODO: maybe make this faster
        for node in clt.iter_descendants():
            if sum((node2.node_id in observed_leaf_ids) for node2 in node.traverse()) == 0:
                node.detach()

        return clt

    def _observe_leaf_with_error(self, leaf):
        """
        Modifies the leaf node in place
        Observes its alleles with error, updates the leaf node's attributes per the actually-observed events
        @return the AlleleList that was observed with errors for this node
        """
        allele_list_with_errors = leaf.allele_list.observe_with_errors(self.error_rate)
        allele_list_with_errors_events = allele_list_with_errors.get_event_encoding(aligner=self.aligner)
        events_per_bcode = [
                [(event.start_pos,
                  event.start_pos + event.del_len,
                  event.insert_str)
                for event in a.events]
            for a in allele_list_with_errors_events]
        leaf.allele_list.process_events(events_per_bcode)
        leaf.allele_events_list = allele_list_with_errors_events
        leaf.sync_allele_events_list_str()
        return allele_list_with_errors

    def observe_leaves(self,
                       sampling_rate: float,
                       cell_lineage_tree: CellLineageTree,
                       give_pruned_clt: bool = True,
                       seed: int = None,
                       bifurcating_only: bool = True,
                       observe_cell_state: bool = True):
        """
        Samples leaves from the cell lineage tree, of those that are not dead

        TODO: this probably won't work very well for very large trees.

        @param sampling_rate: the rate at which alleles from the alive leaf cells are observed
        @param cell_lineage_tree: tree to sample leaves from
        @param seed: controls how the sampling is performed

        @return Tuple with
                    1. a list of the sampled observations (List[ObservedAlignedSeq])
                    2. a cell lineage tree with pruned leaves
        """
        assert (0 < sampling_rate <= 1)
        np.random.seed(seed)
        clt = self._sample_leaves(cell_lineage_tree, sampling_rate)

        # Collapse the tree per ultrametric constraints
        clt = collapsed_tree.collapse_ultrametric(clt)

        if bifurcating_only:
            if clt.is_many_furcating():
                raise ValueError("Need root of clt to be bifurcating")
            for node in clt.traverse():
                if node.is_many_furcating():
                    node.detach()
            for node in clt.get_descendants(strategy="postorder"):
                if len(node.get_children()) == 1:
                    node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

        observations = {}
        # When observing each leaf, observe with specified error rate
        # Gather observed leaves, calculating abundance
        for leaf in clt:
            allele_list_with_errors = self._observe_leaf_with_error(leaf)

            allele_str_id = leaf.allele_events_list_str
            if observe_cell_state:
                cell_id = (allele_str_id, str(leaf.cell_state))
            else:
                cell_id = (allele_str_id,)

            if cell_id in observations:
                observations[cell_id].abundance += 1
            else:
                observations[cell_id] = ObservedAlignedSeq(
                    allele_list=allele_list_with_errors,
                    allele_events_list=leaf.allele_events_list,
                    # TODO: what if lots of possible cell states?!
                    cell_state=leaf.cell_state,
                    abundance=1,
                )

        if len(observations) == 0:
            raise RuntimeError('all lineages extinct, nothing to observe')

        obs_vals = list(observations.values())
        random.shuffle(obs_vals)
        if give_pruned_clt:
            # This section just checks that the collapsing procedure is correct
            obs_evts_list = []
            tree_evts = []
            for o in observations.values():
                obs_evts_list.append(str(o))
            for node in clt.traverse():
                if node.observed:
                    tree_evts.append(node.allele_events_list_str)
            logging.info("diff events? %s", str(set(obs_evts_list) - set(tree_evts)))
            logging.info("diff events? %s", str(set(tree_evts) - set(obs_evts_list)))
            assert set(tree_evts) == set(obs_evts_list), "the two sets are not equal"
            # End of checking
            
            return obs_vals, clt
        else:
            return obs_vals
