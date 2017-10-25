from typing import List
import subprocess

from ete3 import Tree

from clt_observer import ObservedAlignedSeq
from fastq_to_phylip import write_seqs_to_phy
import phylip_parse
from collapsed_tree import CollapsedTree

from constants import MIX_CFG_FILE


class CLTEstimator:
    def estimate(self, observations: List[ObservedAlignedSeq]):
        """
        @return an estimate of the cell lineage tree (post-sampling)
        """
        raise NotImplementedError()

class CLTParsimonyEstimator(CLTEstimator):
    def _process_observations(self, observations: List[ObservedAlignedSeq]):
        processed_seqs = {}
        all_events = set()
        for idx, obs in enumerate(observations):
            evts = obs.barcode.events()
            processed_seqs[idx] = [obs.abundance, evts]
            all_events.update(evts)
        all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}
        return processed_seqs, all_event_dict

    def estimate(self, observations: List[ObservedAlignedSeq]):
        processed_seqs, event_dict = self._process_observations(observations)
        write_seqs_to_phy(processed_seqs, event_dict, "infile", "test.abundance")
        cmd = ["rm -f outfile outtree && mix < mix.cfg"]
        res = subprocess.call(cmd, shell=True)
        assert(res == 0)
        # Parse the outfile -- these are still regular Tree, not CellLineageTrees
        # In the future, we can simultaneously build a cell lineage tree while parsing the
        # output, rather than parsing output and later converting.
        trees = phylip_parse.parse_outfile("outfile")

        # TODO: make this much more efficient - right now checks all other trees
        #       to see if there is an equiv tree.
        uniq_trees = []
        for t in trees:
            collapsed_est_tree = CollapsedTree.collapse(t, preserve_leaves=True)
            if len(uniq_trees) == 0:
                uniq_trees.append(collapsed_est_tree)
            else:
                for uniq_t in uniq_trees:
                    rf_dist = collapsed_est_tree.robinson_foulds(uniq_t)
                    if rf_dist[0] > 0:
                        uniq_trees.append(collapsed_est_tree)

        # Now convert these to FAKEEE cell lineage trees
        #for t in uniq_trees:
        #    clt_new = CellLineageTree(Barcode(), None)
        #    for 
        #    for leaf in par_est_t:
        #        leaf_seq_id = int(leaf.name.replace("seq", ""))
        #        leaf.name = str(obs_leaves[leaf_seq_id].barcode.events())
        return uniq_trees
