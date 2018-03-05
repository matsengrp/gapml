from typing import List, Dict, Tuple
import subprocess
import re
import logging

from ete3 import Tree, TreeNode

from clt_observer import ObservedAlignedSeq
from fastq_to_phylip import write_seqs_to_phy
import phylip_parse
from collapsed_tree import CollapsedTree
from cell_lineage_tree import CellLineageTree
from allele import Allele, AlleleList
from allele_events import Event
from cell_state import CellState
from barcode_metadata import BarcodeMetadata

from constants import MIX_CFG_FILE, MIX_PATH


class CLTEstimator:
    def estimate(self, observations: List[ObservedAlignedSeq]):
        """
        @return an estimate of the cell lineage tree (post-sampling)
        """
        raise NotImplementedError()


class CLTParsimonyEstimator(CLTEstimator):
    def __init__(self, bcode_meta: BarcodeMetadata, out_folder: str, mix_path: str = MIX_PATH):
        self.orig_barcode = bcode_meta.unedited_barcode
        self.bcode_meta = bcode_meta
        self.out_folder = out_folder
        self.mix_path = mix_path

    def _process_observations(self, observations: List[ObservedAlignedSeq]):
        """
        Prepares the observations to be sent to phylip
        Each event is represented by a tuple (start idx, end idx, insertion)

        @return processed_seqs: Dict[str, List[float, List[List[Event]], CellState]]
                    this maps the sequence names to event list and abundance
                all_event_dict: List[Dict[event_tuple, event number]]
                    maps events to their event number
                event_list: List[event_tuple]
                    the reverse of all_event_dict
        """
        # Figure out what events happened
        processed_seqs = {}
        all_events = [set() for _ in range(self.bcode_meta.num_barcodes)]
        for idx, obs in enumerate(observations):
            evts_list = []
            for bcode_idx, allele_evts in enumerate(obs.allele_events_list):
                evts = allele_evts.events
                evts_bcode = [evt for evt in evts]
                all_events[bcode_idx].update(evts_bcode)
                evts_list.append(evts_bcode)
            processed_seqs["seq{}".format(idx)] = [obs.abundance, evts_list, obs.cell_state]
            logging.info("seq%d %s", idx, str(obs))

        # Assemble events in a dictionary
        event_dicts = []
        event_list = []
        num_evts = 0
        for bcode_idx, bcode_evts in enumerate(all_events):
            bcode_evt_list = list(bcode_evts)
            event_list += [(bcode_idx, evt) for evt in bcode_evt_list]
            event_bcode_dict = {evt: num_evts + i for i, evt in enumerate(bcode_evt_list)}
            num_evts += len(event_bcode_dict)
            event_dicts.append(event_bcode_dict)

        return processed_seqs, event_dicts, event_list

    def _do_convert(self,
            clt: CellLineageTree,
            tree: TreeNode,
            event_list: List[Event],
            processed_obs: Dict[str, List[List[Event]]],
            processed_abund: Dict[str, int]):
        """
        Performs the recursive process of forming a cell lineage tree
        """
        for c in tree.children:
            branch_length = c.dist
            child_event_ids = [
                evt_idx
                for evt_idx, allele_char in enumerate(c.binary_allele)
                if allele_char == "1"
            ]
            events = [event_list[idx] for idx in child_event_ids]
            grouped_events = [[] for _ in range(self.bcode_meta.num_barcodes)]
            for evt in events:
                bcode_idx = evt[0]
                actual_evt = evt[1]
                grouped_events[bcode_idx].append(actual_evt)

            child_allele_list = AlleleList(
                    [self.orig_barcode] * self.bcode_meta.num_barcodes,
                    self.bcode_meta)
            child_allele_list.process_events(
                    [[(event.start_pos,
                        event.del_end,
                        event.insert_str) for event in events] for events in grouped_events])
            cell_state = None if not c.is_leaf() else processed_obs[c.name]
            cell_abundance = 0 if not c.is_leaf() else processed_abund[c.name]
            child_clt = CellLineageTree.convert(c,
                                        allele_list=child_allele_list,
                                        cell_state=cell_state,
                                        abundance=cell_abundance,
                                        dist=branch_length)
            clt.add_child(child_clt)
            self._do_convert(child_clt, c, event_list, processed_obs, processed_abund)

    def convert_tree_to_clt(self,
            tree: TreeNode,
            event_list: List[Event],
            processed_obs: Dict[str, List[List[Event]]],
            processed_abund: Dict[str, int]):
        """
        Make a regular TreeNode to a Cell lineage tree
        """
        # TODO: update cell state maybe in the future?
        clt = CellLineageTree.convert(
                tree,
                AlleleList(
                    [self.orig_barcode] * self.bcode_meta.num_barcodes,
                    self.bcode_meta),
                cell_state=None)
        self._do_convert(clt, tree, event_list, processed_obs, processed_abund)
        return clt

    def _create_mix_cfg(self):
        mix_cfg_lines = [
                "%s/infile\n" % self.out_folder,
                "F\n",
                "%s/outfile\n" % self.out_folder,
                ]
        with open(MIX_CFG_FILE, "r") as f:
            mix_cfg_lines += f.readlines()
        new_mix_cfg_file = "%s/%s" % (self.out_folder, MIX_CFG_FILE)
        with open(new_mix_cfg_file, "w") as f:
            for line in mix_cfg_lines:
                f.write(line)
        return new_mix_cfg_file

    def estimate(self,
            observations: List[ObservedAlignedSeq],
            encode_hidden: bool = True,
            use_cell_state: bool = False,
            max_uniq_trees: int = None):
        """
        @return a list of unique cell lineage tree estimates
                calls out to mix on the command line
                writes files: infile, test.abundance, outfile

        TODO: send weights to phylip mix too eventually
        """
        processed_seqs, event_dicts, event_list = self._process_observations(
            observations)
        new_mix_cfg_file = self._create_mix_cfg()
        infile = "%s/infile" % self.out_folder
        abundance_file = "%s/test.abundance" % self.out_folder
        write_seqs_to_phy(
                processed_seqs,
                event_dicts,
                infile,
                abundance_file,
                encode_hidden=encode_hidden,
                use_cell_state=use_cell_state)
        outfile = "%s/outfile" % self.out_folder
        cmd = ["rm -f %s && %s < %s" % (
            outfile,
            self.mix_path,
            new_mix_cfg_file)]
        res = subprocess.call(cmd, shell=True)
        # Check that mix ran properly
        assert res == 0, "Mix failed to run"

        # Parse the outfile -- these are still regular Tree, not CellLineageTrees
        # In the future, we can simultaneously build a cell lineage tree while parsing the
        # output, rather than parsing output and later converting.
        trees = phylip_parse.parse_outfile(outfile, abundance_file)

        # Only return unique trees, so check if trees are equivalent by first
        # collapsing them to get multifurcating trees
        # TODO: make this much more efficient - right now checks all other trees
        #       to see if there is an equiv tree.
        uniq_trees = []
        for t in trees:
            collapsed_est_tree = CollapsedTree.collapse_zero_lens(t)
            if len(uniq_trees) == 0:
                uniq_trees.append(collapsed_est_tree)
            else:
                # We are going to use the unrooted tree assuming that the collapsed tree output
                # does not have multifurcating branches...
                dists = [
                    collapsed_est_tree.robinson_foulds(
                        uniq_t,
                        unrooted_trees=True)[0]
                    for uniq_t in uniq_trees]
                if min(dists) > 0:
                    uniq_trees.append(collapsed_est_tree)
                    if max_uniq_trees is not None and len(uniq_trees) > max_uniq_trees:
                        break
                    else:
                        continue

        # print('trees post-collapse: {}'.format(len(uniq_trees)))

        # Get a mapping from cell to cell state
        processed_obs = {k: v[2] for k, v in processed_seqs.items()}
        # Get a mapping from cell to abundance
        processed_abund = {k: v[0] for k, v in processed_seqs.items()}
        # Now convert these trees to CLTs
        uniq_clts = []
        for t in uniq_trees:
            clt_new = self.convert_tree_to_clt(t, event_list, processed_obs, processed_abund)
            clt_new.label_tree_with_strs()
            uniq_clts.append(clt_new)
        return uniq_clts
