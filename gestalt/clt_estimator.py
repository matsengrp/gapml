from typing import List, Dict, Tuple
import subprocess
import re
import logging
import random

from ete3 import Tree, TreeNode

from clt_observer import ObservedAlignedSeq
from fastq_to_phylip import write_seqs_to_phy
import phylip_parse
import collapsed_tree
from cell_lineage_tree import CellLineageTree
from allele import Allele, AlleleList
from allele_events import Event
from cell_state import CellState
from barcode_metadata import BarcodeMetadata
from tree_distance import UnrootRFDistanceMeasurer

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
        self.outfile = "%s/outfile" % self.out_folder
        self.infile = "%s/infile" % self.out_folder
        self.abundance_file = "%s/test.abundance" % self.out_folder
        self.dist_measurer = UnrootRFDistanceMeasurer(None, None)

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
        for node in clt.traverse():
            node.add_feature("observed", node.is_leaf())

        while len(clt.get_children()) == 1:
            child_node = clt.get_children()[0]
            child_node.delete(prevent_nondicotomic=True, preserve_branch_length=True)
        assert(clt.is_root())

        return clt

    def _create_mix_cfg(self, seed_num, num_jumbles=5):
        mix_cfg_lines = [
                self.infile,
                "F",
                self.outfile,
                "1",
                "4",
                "5",
                "6",
                "P",
                "J",
                # Seed for jumbling must be odd
                str(seed_num * 2 + 1),
                str(num_jumbles),
                "Y"
                ]
        mix_cfg_lines = ["%s\n" % s for s in mix_cfg_lines]
        new_mix_cfg_file = "%s/%s" % (self.out_folder, MIX_CFG_FILE)
        with open(new_mix_cfg_file, "w") as f:
            for line in mix_cfg_lines:
                f.write(line)
        return new_mix_cfg_file

    def estimate(self,
            observations: List[ObservedAlignedSeq],
            encode_hidden: bool = True,
            use_cell_state: bool = False,
            do_collapse: bool = False,
            num_mix_runs: int = 2):
        """
        @param observations: the observations to run MIX on
        @param encode_hidden: indicate hidden states to MIX
        @param use_cell_state: ignored -- this is always false
        @param num_mix_runs: number of mix runs with different seeds
        @param do_collapse: return collapsed trees instead of bifurcating trees

        @return a list of unique cell lineage tree estimates
                calls out to mix on the command line
                writes files: infile, test.abundance, outfile

        TODO: send weights to phylip mix too eventually
        """
        processed_seqs, event_dicts, event_list = self._process_observations(
            observations)

        write_seqs_to_phy(
                processed_seqs,
                event_dicts,
                self.infile,
                self.abundance_file,
                encode_hidden=encode_hidden,
                use_cell_state=use_cell_state)

        # Now run mix many times
        tree_lists = []
        for seed_i in range(num_mix_runs):
            new_mix_cfg_file = self._create_mix_cfg(seed_i)
            pars_trees = self.run_mix(new_mix_cfg_file)
            # Note: These trees aren't necessarily unique
            tree_lists.append(pars_trees)

        # Read out the results
        bifurcating_trees = [t for trees in tree_lists for t in trees]
        logging.info("num bifurcating trees %d", len(bifurcating_trees))
        if do_collapse:
            # Collapse trees
            collapsed_trees = [collapsed_tree.collapse_zero_lens(t) for t in bifurcating_trees]
            # Get the unique collapsed trees
            mix_trees = self.dist_measurer.get_uniq_trees(collapsed_trees)
        else:
            mix_trees = bifurcating_trees

        # Get a mapping from cell to cell state
        processed_obs = {k: v[2] for k, v in processed_seqs.items()}
        # Get a mapping from cell to abundance
        processed_abund = {k: v[0] for k, v in processed_seqs.items()}
        # Now convert these trees to CLTs
        clts = []
        for t in mix_trees:
            clt_new = self.convert_tree_to_clt(t, event_list, processed_obs, processed_abund)
            clt_new.label_tree_with_strs()
            clts.append(clt_new)
        return clts

    def run_mix(self, new_mix_cfg_file):
        """
        Run mix once with the mix config file
        @return trees from mix
        """
        # Clean up the outfile (and potentially other files)
        # because mix may get confused otherwise
        cmd = ["rm -f %s && %s < %s" % (
            self.outfile,
            self.mix_path,
            new_mix_cfg_file)]
        res = subprocess.call(cmd, shell=True)
        # Check that mix ran properly
        assert res == 0, "Mix failed to run"

        # Parse the outfile -- these are still regular Tree, not CellLineageTrees
        # In the future, we can simultaneously build a cell lineage tree while parsing the
        # output, rather than parsing output and later converting.
        pars_trees = phylip_parse.parse_outfile(self.outfile, self.abundance_file)

        # Clean up the outfile because it is really big and takes up
        # disk space
        cmd = ["rm -f %s" % self.outfile]
        res = subprocess.call(cmd, shell=True)
        # Check that clean up happened
        assert res == 0, "clean up happened"
        return pars_trees
