from typing import List
import numpy as np
import logging

from allele import AlleleList
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

    def set_allele_list(self, allele_list: AlleleList):
        self.allele_list = allele_list
        self.allele_events_list = allele_list.get_event_encoding()

    def get_allele_str(self):
        return "||".join([str(allele_evts) for allele_evts in self.allele_events_list])

    def __str__(self):
        return "%s,%s" % (self.get_allele_str(), str(self.cell_state))

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
                is_sampled = np.random.uniform() < sampling_rate
                if is_sampled:
                    #logging.info("sampled id %d" % leaf.node_id)
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
        collapsed_tree._remove_single_child_unobs_nodes(clt)
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
                       seed: int = None,
                       observe_cell_state: bool = False):
        """
        Samples leaves from the cell lineage tree, of those that are not dead
        Note: a bit slow for big trees

        @param sampling_rate: the rate at which alleles from the alive leaf cells are observed
        @param cell_lineage_tree: tree to sample leaves from
        @param seed: controls how the sampling is performed
        @param observe_cell_state: whether or not to record the cell state

        @return Tuple with
                    1. a list of the sampled observations (List[ObservedAlignedSeq])
                    2. the subtree of the full cell lineage tree with the sampled leaves
                    3. List[leaf node ids that observation at this index maps to]
        """
        assert (0 < sampling_rate <= 1)
        np.random.seed(seed)
        sampled_clt = self._sample_leaves(cell_lineage_tree, sampling_rate)
        sampled_clt.label_tree_with_strs()

        observations = {}
        # When observing each leaf, observe with specified error rate
        # Gather observed leaves, calculating abundance
        for leaf in sampled_clt:
            allele_list_with_errors = self._observe_leaf_with_error(leaf)
            allele_events_list_with_errors = allele_list_with_errors.get_event_encoding()

            if observe_cell_state:
                collapse_id = (leaf.allele_events_list_str, str(leaf.cell_state))
            else:
                collapse_id = leaf.allele_events_list_str

            if collapse_id in observations:
                observations[collapse_id][0].abundance += 1
                observations[collapse_id][1].append(leaf.node_id)
            else:
                obs_seq = ObservedAlignedSeq(
                    allele_list=allele_list_with_errors,
                    allele_events_list=allele_events_list_with_errors,
                    cell_state=leaf.cell_state if observe_cell_state else None,
                    abundance=1,
                )
                observations[collapse_id] = (obs_seq, [leaf.node_id])

        if len(observations) == 0:
            raise RuntimeError('all lineages extinct, nothing to observe')

        obs_vals = [obs[0] for obs in observations.values()]
        obs_idx_to_leaves = [obs[1] for obs in observations.values()]

        return obs_vals, sampled_clt, obs_idx_to_leaves
