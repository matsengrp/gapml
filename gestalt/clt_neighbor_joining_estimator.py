import logging
import time
import numpy as np
import subprocess
from ete3 import Tree
from typing import List

from clt_observer import ObservedAlignedSeq
from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
import ancestral_events_finder
from collapsed_tree import _remove_single_child_unobs_nodes
from barcode_metadata import BarcodeMetadata

from Bio.Phylo import draw_ascii, write
from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor

class CLTNeighborJoiningEstimator(CLTEstimator):
    def __init__(
            self,
            bcode_meta: BarcodeMetadata,
            scratch_dir: str,
            obs_leaves: List[ObservedAlignedSeq]):
        self.bcode_meta = bcode_meta
        self.scratch_dir = scratch_dir
        self.obs_leaves = obs_leaves

    def estimate(self):
        """
        Actually runs neighbor joining
        @return CellLineageTree where the dist on each node corresponds
                to the number of different events
        """
        # construct a right-triangular distance matrix using the cardinality of the
        # symmetric difference of the sets of events in each barcode in two leaves
        distance_matrix = []
        for row, row_leaf in enumerate(self.obs_leaves):
            distance_matrix_row = []
            for col_leaf in self.obs_leaves[:(row + 1)]:
                distance_matrix_row.append(sum(len(set(row_allele_events.events) ^ set(col_allele_events.events))
                                               for row_allele_events, col_allele_events in zip(row_leaf.allele_events_list,
                                                                                               col_leaf.allele_events_list)))
            distance_matrix.append(distance_matrix_row)
        distance_matrix = _DistanceMatrix(names=[str(i) for i in range(len(self.obs_leaves))],
                                          matrix=distance_matrix)
        constructor = DistanceTreeConstructor()
        fitted_tree = constructor.nj(distance_matrix)
        newick_tree_file = '{}/tmp.nk'.format(self.scratch_dir)
        write(fitted_tree, newick_tree_file, 'newick')

        # Read fitted tree into ETE tree
        fitted_tree = Tree(newick_tree_file, format=1)
        # Convert the fitted tree back to a cell lineage tree
        # NOTE: arbitrarily using the first allele in observed leaves to initialize
        #       barcode states. We will later update the leaf states only
        root_clt = CellLineageTree(
                self.obs_leaves[0].allele_list,
                self.obs_leaves[0].allele_events_list,
                self.obs_leaves[0].cell_state,
                dist=0)
        CLTNeighborJoiningEstimator._do_convert(fitted_tree, root_clt)
        # update the leaves to have the correct barcode states
        for leaf in root_clt:
            leaf_parent = leaf.up
            leaf.detach()
            leaf_parent.add_child(CellLineageTree(
                    self.obs_leaves[int(leaf.name)].allele_list,
                    self.obs_leaves[int(leaf.name)].allele_events_list,
                    self.obs_leaves[int(leaf.name)].cell_state,
                    dist=leaf.dist,
                    abundance=self.obs_leaves[int(leaf.name)].abundance))
        for leaf in root_clt:
            leaf.add_feature('leaf_key', leaf.allele_events_list_str)
        logging.info("Done with fitting tree using neighbor joining")
        logging.info(fitted_tree.get_ascii(attributes=["dist"]))

        return root_clt



    @staticmethod
    def _do_convert(tree_node, clt_node):
        for child in tree_node.get_children():
            clt_child = CellLineageTree(
                clt_node.allele_list,
                clt_node.allele_events_list,
                clt_node.cell_state,
                dist=child.dist)
            clt_child.name = child.name
            clt_node.add_child(clt_child)
            CLTNeighborJoiningEstimator._do_convert(child, clt_child)
