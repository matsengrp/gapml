from typing import List
import numpy as np
import logging

from allele import AlleleList, Allele
from allele_events import AlleleEvents, Event
from cell_state import CellState
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
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
            bcode_meta: BarcodeMetadata,
            error_rate: float = 0,
            aligner: Aligner = None):
        """
        @param error_rate: sequencing error, introduce alternative bases uniformly at this rate
        """
        self.bcode_meta = bcode_meta
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
    def sample_leaves(clt_orig: CellLineageTree, sampling_rate: float):
        """
        Makes a copy of the original tree, samples leaves, and returns back a new tree
        @return CellLineageTree that represents a subsampled `clt_orig`
        """
        observed_leaf_ids = CLTObserver._sample_leaf_ids(clt_orig, sampling_rate)
        sampled_clt = CellLineageTree.prune_tree(clt_orig, observed_leaf_ids)
        collapsed_tree._remove_single_child_unobs_nodes(sampled_clt)
        sampled_clt.label_node_ids()
        return sampled_clt

    def _observe_leaf_with_error(self, leaf, error=0):
        """
        Modifies the leaf node in place
        Observes its alleles with error, updates the leaf node's attributes per the actually-observed events
        @return the AlleleList that was observed with errors for this node
        """
        allele_list_with_errors = leaf.allele_list.observe_with_errors(error_rate=error)
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
                       sampled_clt: CellLineageTree,
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
        np.random.seed(seed)
        sampled_clt.label_tree_with_strs()
        observations = {}
        # When observing each leaf, observe with specified error rate
        # Gather observed leaves, calculating abundance
        for leaf in sampled_clt:
            allele_list_with_errors = self._observe_leaf_with_error(leaf, error=0)
            allele_events_list_with_errors = allele_list_with_errors.get_event_encoding()

            # TODO: i dont think this observes cell state correctly
            # But then again, all te simulations with cell state are broekn probably
            if observe_cell_state:
                collapse_id = (leaf.allele_events_list_str, str(leaf.cell_state))
            else:
                collapse_id = leaf.allele_events_list_str

            if collapse_id in observations:
                observations[collapse_id][0].abundance += 1
                observations[collapse_id][1].append(leaf.node_id)
                observations[collapse_id][2].append(leaf)
            else:
                obs_seq = ObservedAlignedSeq(
                    allele_list=allele_list_with_errors,
                    allele_events_list=allele_events_list_with_errors,
                    cell_state=leaf.cell_state if observe_cell_state else None,
                    abundance=1,
                )
                observations[collapse_id] = (obs_seq, [leaf.node_id], [leaf])

        if len(observations) == 0:
            raise RuntimeError('all lineages extinct, nothing to observe')

        if self.error_rate > 0:
            self._make_errors(observations)

        obs_vals = [obs[0] for obs in observations.values()]
        obs_idx_to_leaves = [obs[1] for obs in observations.values()]

        return obs_vals, obs_idx_to_leaves

    def _make_errors(self, observations):
        # Introduce errors when observing the indel

        # Map a random subset of indels to an indel that is slightly wrong
        all_error_maps = []
        for i in range(self.bcode_meta.num_barcodes):
            unique_events = set([evt for obs_seq, _, _ in observations.values() for evt in obs_seq.allele_events_list[i].events])
            error_map = {}
            all_error_maps.append(error_map)
            for evt in unique_events:
                insert_str_perturb = evt.insert_str
                if np.random.rand() < self.error_rate:
                    if np.random.rand() < 0.5:
                        insert_str_perturb = evt.insert_str[:evt.insert_len//2]
                    else:
                        insert_str_perturb = evt.insert_str + evt.insert_str[:evt.insert_len//2]

                start_pos_perturb = evt.start_pos
                if np.random.rand() < self.error_rate:
                    # TODO: Be careful about deactivating other targets
                    min_start = self.bcode_meta.abs_cut_sites[evt.min_target - 1] if evt.min_target > 0 else 0
                    max_start = self.bcode_meta.abs_cut_sites[evt.min_target]
                    start_pos_perturb = np.random.randint(min_start, max_start)

                del_len_perturb = evt.del_end - start_pos_perturb
                if np.random.rand() < self.error_rate:
                    # TODO: Be careful about deactivating other targets
                    min_end = self.bcode_meta.abs_cut_sites[evt.max_target]
                    max_end = self.bcode_meta.abs_cut_sites[evt.max_target + 1] if evt.max_target < self.bcode_meta.n_targets - 1 else self.bcode_meta.orig_length - 1
                    end_len_perturb = np.random.randint(min_end, max_end)
                    del_len_perturb = end_len_perturb - start_pos_perturb

                assert start_pos_perturb >= 0
                error_with_evt = Event(
                        start_pos_perturb,
                        del_len_perturb,
                        evt.min_target,
                        evt.max_target,
                        insert_str=insert_str_perturb)
                error_map[evt] = error_with_evt

        # Update the observations
        for i in range(self.bcode_meta.num_barcodes):
            for k, (obs_seq, _, _) in observations.items():
                allele = obs_seq.allele_list.alleles[i]
                allele_evts = obs_seq.allele_events_list[i]
                any_diffs = any([evt != all_error_maps[i][evt] for evt in allele_evts.events])
                if any_diffs:
                    perturbed_allele = Allele(self.bcode_meta.unedited_barcode, self.bcode_meta)
                    my_evts = [all_error_maps[i][evt] for evt in allele_evts.events]
                    perturbed_allele.process_events([(evt.start_pos, evt.del_end, evt.insert_str) for evt in my_evts])
                    alleles = obs_seq.allele_list.alleles[:i] + [perturbed_allele] + obs_seq.allele_list.alleles[i + 1:]
                    allele_list = AlleleList(
                            [a.allele for a in alleles],
                            self.bcode_meta)
                    obs_seq.set_allele_list(allele_list)

        # Label the true subtree with the error leaves
        for k, (obs_seq, _, nodes) in observations.items():
            for node in nodes:
                node.add_feature(
                    "allele_events_list_error",
                    obs_seq.allele_events_list)
                node.add_feature(
                    "allele_list_error",
                    obs_seq.allele_list)
