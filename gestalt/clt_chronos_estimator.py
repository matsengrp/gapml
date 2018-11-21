import logging
import time
import numpy as np
import subprocess
from ete3 import Tree

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
import ancestral_events_finder
from collapsed_tree import _remove_single_child_unobs_nodes
from barcode_metadata import BarcodeMetadata

class CLTChronosEstimator(CLTEstimator):
    def __init__(
            self,
            tree: CellLineageTree,
            bcode_meta: BarcodeMetadata,
            scratch_dir: str,
            tot_time: float):
        self.tree = tree
        self.bcode_meta = bcode_meta
        self.scratch_dir = scratch_dir
        self.tot_time = tot_time

    def estimate(
            self,
            lam: float):
        """
        Actually runs chronos to fit tree branch lengths.
        @return CellLineageTree where the dist on each node corresponds
                to the estimated branch length
        """
        tree = self.tree.copy()
        _remove_single_child_unobs_nodes(tree)
        if len(tree.get_children()) == 1:
            tree.get_children()[0].delete()
        ancestral_events_finder.annotate_ancestral_states(tree, self.bcode_meta)
        ancestral_events_finder.get_parsimony_score(tree)
        logging.info(tree.get_ascii(attributes=["dist"]))
        print(tree.get_ascii(attributes=["dist"]))
        print(tree)

        # Track the leaves by naming them.
        # So we can reconstruct the fitted tree from the R output
        orig_node_dict = {}
        for node_num, node in enumerate(tree.traverse()):
            node.name = str(node_num)
            orig_node_dict[node.name] = node

        # Write tree to newick
        suffix = "%d%d" % (int(time.time()), np.random.randint(1000000))
        tree_in_file = "%s/tree_newick%s.txt" % (
                self.scratch_dir, suffix)
        with open(tree_in_file, "w") as f:
            f.write(tree.write(format=3))
            f.write("\n")

        tree_out_file = "%s/fitted_tree%s.txt" % (
                self.scratch_dir, suffix)

        cmd = [
                'Rscript',
                '../R/fit_chronos.R',
                tree_in_file,
                tree_out_file,
                str(self.tot_time),
                str(lam),
        ]

        print("Calling:", " ".join(cmd))
        res = subprocess.check_output(cmd)
        print("resss", res)

        # Read fitted tree
        with open(tree_out_file, "r") as f:
            newick_tree = f.readlines()[0]
            fitted_tree = Tree(newick_tree, format=3)
        logging.info("Done with fitting tree using chronos, lam %f", lam)
        logging.info(fitted_tree.get_ascii(attributes=["dist"]))
        logging.info(fitted_tree.get_ascii(attributes=["name"]))

        # Convert the fitted tree back to a cell lineage tree
        root_clt = CellLineageTree(
                tree.allele_list,
                tree.allele_events_list,
                tree.cell_state,
                dist=0)
        root_clt.add_feature('node_id', tree.node_id)
        CLTChronosEstimator._do_convert(fitted_tree, root_clt, orig_node_dict)

        for leaf in root_clt:
            leaf_parent = leaf.up
            leaf.detach()
            orig_cell_lineage_tree = orig_node_dict[leaf.name]
            orig_cell_lineage_tree.dist = leaf.dist
            orig_cell_lineage_tree.detach()
            leaf_parent.add_child(orig_cell_lineage_tree)

        return root_clt

    @staticmethod
    def _do_convert(tree_node, clt_node, orig_node_dict):
        for child in tree_node.get_children():
            clt_child = CellLineageTree(
                clt_node.allele_list,
                clt_node.allele_events_list,
                clt_node.cell_state,
                dist=child.dist)
            clt_child.name = child.name
            clt_child.node_id = orig_node_dict[child.name].node_id
            clt_node.add_child(clt_child)
            CLTChronosEstimator._do_convert(child, clt_child, orig_node_dict)
