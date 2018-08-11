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
from allele import Allele, AlleleList, AlleleEvents
from allele_events import Event
from cell_state import CellState
from barcode_metadata import BarcodeMetadata
from tree_distance import UnrootRFDistanceMeasurer
import data_binarizer

from constants import MIX_CFG_FILE
from constant_paths import MIX_PATH


class CLTEstimator:
    def estimate(self, observations: List[ObservedAlignedSeq]):
        """
        @return an estimate of the cell lineage tree (post-sampling)
        """
        raise NotImplementedError()

class CLTParsimonyEstimator(CLTEstimator):
    def __init__(self, bcode_meta: BarcodeMetadata, out_folder: str, mix_path: str = MIX_PATH):
        """
        @param out_folder: file path to where to put the mix input and output files
        @param mix_path: path for running mix
        """
        self.orig_barcode = bcode_meta.unedited_barcode
        self.bcode_meta = bcode_meta
        self.out_folder = out_folder
        self.mix_path = mix_path
        self.outfile = "%s/outfile" % self.out_folder
        self.infile = "%s/infile" % self.out_folder
        assert len(self.infile) < 200, 'Mix is super picky about string lengths'
        self.abundance_file = "%s/test.abundance" % self.out_folder
        self.dist_measurer = UnrootRFDistanceMeasurer(None, None)

    def _do_convert(self,
            clt: CellLineageTree,
            tree: TreeNode,
            event_list: List[Event],
            processed_abund: Dict[str, int]):
        """
        Performs the recursive process of forming a cell lineage tree
        Converts children nodes of `tree` and appends them to `clt`

        @param clt: CellLineageTree of the current node, this is the thing we are creating
        @param tree: TreeNode of the current node, this is the thing we are converting from
        @param event_list: the list of events observed in all the data, used to reconstruct nodes
        @param processed_abund: dictionary mapping node name to abundance values
        """
        for c in tree.children:
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

            allele_events_list = []
            for grp_evts in grouped_events:
                sorted_grp_evts = sorted(grp_evts, key=lambda evt:evt.start_pos)
                final_obs_events = sorted_grp_evts[:1]
                for evt in sorted_grp_evts[1:]:
                    if not final_obs_events[-1].hides(evt):
                        final_obs_events.append(evt)
                allele_events_list.append(
                        AlleleEvents(final_obs_events))

            cell_abundance = 0 if not c.is_leaf() else processed_abund[c.name]
            child_clt = CellLineageTree.convert(c,
                                        allele_events_list=allele_events_list,
                                        abundance=cell_abundance,
                                        dist=c.dist)
            clt.add_child(child_clt)
            self._do_convert(child_clt, c, event_list, processed_abund)

    def convert_tree_to_clt(self,
            tree: TreeNode,
            event_list: List[Event],
            processed_abund: Dict[str, int]):
        """
        Make a regular TreeNode to a Cell lineage tree
        """
        clt = CellLineageTree.convert(
                tree,
                allele_events_list=[AlleleEvents([]) for _ in range(self.bcode_meta.num_barcodes)])
        self._do_convert(clt, tree, event_list, processed_abund)

        return clt

    def _create_mix_cfg(self, seed_num, num_jumbles=5):
        """
        Writes a mix config file
        TODO: create and specify the weights file too, weights based on abundance values

        @param seed_num: seed passed to mix, specified in MIX config
        @param num_jumbles: number of jumbles, specified in MIX config
        @return file path to the mix config file
        """
        assert seed_num % 2 == 1
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
                str(seed_num),
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
            mix_seed: int = 1,
            num_jumbles: int = 5):
        """
        @param observations: the observations to run MIX on
        @param encode_hidden: indicate hidden states to MIX
        @param use_cell_state: ignored -- this is always false

        @return a list of unique cell lineage tree estimates
                calls out to mix on the command line
                writes files: infile, test.abundance, outfile

        TODO: send weights to phylip mix too eventually
        """
        processed_seqs, event_dicts, event_list = data_binarizer.binarize_observations(
            self.bcode_meta,
            observations)

        write_seqs_to_phy(
                processed_seqs,
                event_dicts,
                self.infile,
                self.abundance_file,
                encode_hidden=encode_hidden,
                use_cell_state=use_cell_state)

        new_mix_cfg_file = self._create_mix_cfg(mix_seed, num_jumbles)
        bifurcating_trees = self.run_mix(new_mix_cfg_file)

        # Read out the results
        logging.info("num bifurcating trees %d", len(bifurcating_trees))

        # Get a mapping from cell to abundance
        processed_abund = {k: v[0] for k, v in processed_seqs.items()}
        # Now convert these trees to CLTs
        clts = []
        for t in bifurcating_trees:
            clt_new = self.convert_tree_to_clt(t, event_list, processed_abund)
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
        print(cmd)
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
