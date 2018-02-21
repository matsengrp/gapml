from typing import List, Dict, Tuple
import subprocess
import re

from ete3 import Tree, TreeNode

from clt_observer import ObservedAlignedSeq
from fastq_to_phylip import write_seqs_to_phy
import phylip_parse
from collapsed_tree import CollapsedTree
from cell_lineage_tree import CellLineageTree
from allele import Allele
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
    def __init__(self, orig_barcode: List[str], bcode_meta: BarcodeMetadata, mix_path: str = MIX_PATH):
        self.orig_barcode = orig_barcode
        self.bcode_meta = bcode_meta
        self.mix_path = mix_path

    def _process_observations(self, observations: List[ObservedAlignedSeq]):
        """
        Prepares the observations to be sent to phylip
        Each event is represented by a tuple (start idx, end idx, insertion)

        @return processed_seqs: Dict[str, List[float, List[event_tuple], CellState]]
                    this maps the sequence names to event list and abundance
                all_event_dict: Dict[event_tuple, event number]
                    maps events to their event number
                event_list: List[event_tuple]
                    the reverse of all_event_dict
        """
        processed_seqs = {}
        all_events = set()
        for idx, obs in enumerate(observations):
            evts = obs.allele_events.events
            processed_seqs["seq{}".format(idx)] = [obs.abundance, evts, obs.cell_state]
            all_events.update(evts)
        all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}
        event_list = [event_id for i, event_id in enumerate(all_events)]
        return processed_seqs, all_event_dict, event_list

    def _do_convert(self,
            clt: CellLineageTree,
            tree: TreeNode,
            event_list: List[Tuple[int, int, str]],
            processed_obs: Dict[str, CellState],
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
            child_allele = Allele(self.orig_barcode, self.bcode_meta)
            child_allele.process_events([(event.start_pos,
                                         event.start_pos + event.del_len,
                                         event.insert_str) for event in events])
            cell_state = None if not c.is_leaf() else processed_obs[c.name]
            cell_abundance = 0 if not c.is_leaf() else processed_abund[c.name]
            child_clt = CellLineageTree(child_allele,
                                        cell_state=cell_state,
                                        abundance=cell_abundance,
                                        dist=branch_length)

            clt.add_child(child_clt)
            self._do_convert(child_clt, c, event_list, processed_obs, processed_abund)

    def convert_tree_to_clt(self,
            tree: TreeNode,
            event_list: List[Tuple[int, int, str]],
            processed_obs: Dict[str, CellState],
            processed_abund: Dict[str, int]):
        """
        Make a regular TreeNode to a Cell lineage tree
        """
        # TODO: update cell state maybe in the future?
        clt = CellLineageTree(
                Allele(self.orig_barcode, self.bcode_meta),
                cell_state=None)
        self._do_convert(clt, tree, event_list, processed_obs, processed_abund)
        return clt

    def estimate(self, observations: List[ObservedAlignedSeq],
                 encode_hidden: bool = False):
        """
        @return a list of unique cell lineage tree estimates
                calls out to mix on the command line
                writes files: infile, test.abundance, outfile, outtree

        TODO: have these input/output files be written in a tmp folder instead
        """
        processed_seqs, event_dict, event_list = self._process_observations(
            observations)
        write_seqs_to_phy(processed_seqs, event_dict, "infile",
                          "test.abundance", encode_hidden=encode_hidden)
        cmd = ["rm -f outfile outtree && %s < mix.cfg" % self.mix_path]
        res = subprocess.call(cmd, shell=True)
        assert (res == 0)
        # Parse the outfile -- these are still regular Tree, not CellLineageTrees
        # In the future, we can simultaneously build a cell lineage tree while parsing the
        # output, rather than parsing output and later converting.
        trees = phylip_parse.parse_outfile("outfile", "test.abundance")

        # print('trees pre-collapse: {}'.format(len(trees)))

        # Only return unique trees, so check if trees are equivalent by first
        # collapsing them to get multifurcating trees
        # TODO: make this much more efficient - right now checks all other trees
        #       to see if there is an equiv tree.
        uniq_trees = []
        for t in trees:
            collapsed_est_tree = CollapsedTree.collapse(t, collapse_zero_lens=True)
            if len(uniq_trees) == 0:
                uniq_trees.append(collapsed_est_tree)
            else:
                # We are going to use the unrooted tree assuming that the collapsed tree output
                # does not have multifurcating branches...
                dists = [
                    collapsed_est_tree.robinson_foulds(uniq_t, unrooted_trees=True)[0]
                    for uniq_t in uniq_trees]
                if min(dists) > 0:
                    uniq_trees.append(collapsed_est_tree)
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
            uniq_clts.append(clt_new)
        return uniq_clts
