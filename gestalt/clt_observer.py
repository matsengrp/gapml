from typing import List, Dict, Tuple
import numpy as np
import logging

from allele import Allele, AlleleList
from allele_events import AlleleEvents
from cell_state import CellState
from cell_lineage_tree import CellLineageTree
from alignment import Aligner
from collapsed_tree import CollapsedTree

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

    def observe_leaves(self,
                       sampling_rate: float,
                       cell_lineage_tree: CellLineageTree,
                       give_pruned_clt: bool = True,
                       seed: int = None,
                       observe_cell_state: bool = True):
        """
        Samples leaves from the cell lineage tree, of those that are not dead

        TODO: this probably won't work very well for very large trees.

        @param sampling_rate: the rate at which alleles from the alive leaf cells are observed
        @param cell_lineage_tree: tree to sample leaves from
        @param seed: controls how the sampling is performed

        @return a list of the sampled observations (List[ObservedAlignedSeq])
                a cell lineage tree with pruned leaves
        """
        assert (0 < sampling_rate <= 1)
        np.random.seed(seed)
        clt = cell_lineage_tree.copy()

        # Stores only leaf names!
        observed_leaves = set()
        # First sample the leaves
        for leaf in clt:
            if not leaf.dead:
                is_sampled = np.random.binomial(1, sampling_rate)
                if is_sampled:
                    observed_leaves.add(leaf.name)

        # Now prune the tree -- custom pruning (cause ete was doing weird things...)
        # TODO: maybe make this faster
        for node in clt.iter_descendants():
            if sum((node2.name in observed_leaves) for node2 in node.traverse()) == 0:
                node.detach()

        observations = {}
        # Now gather the observed leaves, calculating abundance
        for leaf in clt:
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
            # TODO: this is a bit tricky -- how does parsimony work if we want to recover
            #       the collapsed branch length by the other idx?
            allele_str_id = tuple([str(a) for a in allele_list_with_errors_events])
            if observe_cell_state:
                cell_id = (allele_str_id, str(leaf.cell_state))
            else:
                cell_id = (allele_str_id,)

            if cell_id in observations:
                observations[cell_id].abundance += 1
            else:
                observations[cell_id] = ObservedAlignedSeq(
                    allele_list=allele_list_with_errors,
                    allele_events_list=allele_list_with_errors_events,
                    # TODO: what if lots of possible cell states?!
                    cell_state=leaf.cell_state,
                    abundance=1,
                )

        if len(observations) == 0:
            raise RuntimeError('all lineages extinct, nothing to observe')

        if give_pruned_clt:
            # Collapse the tree
            clt = CollapsedTree.collapse_ultrametric(clt)
            obs_evts_list = []
            tree_evts = []
            for o in observations.values():
                obs_evts_list.append(str(o))
            print("ooo", len(set(obs_evts_list)))
            for node in clt.traverse():
                if node.observed:
                    tree_evts.append(node.allele_events_list_str)
            logging.info("diff events? %s", str(set(obs_evts_list) - set(tree_evts)))
            print(clt.get_ascii(attributes=["observed"], show_internal=True))
            print(set(tree_evts) - set(obs_evts_list))
            print(set(obs_evts_list) - set(tree_evts))
            assert set(tree_evts) == set(obs_evts_list), "the two sets are not equal"
            return list(observations.values()), clt
        else:
            return list(observations.values())
