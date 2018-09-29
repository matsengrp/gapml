import re
from ete3 import TreeNode
from typing import List, Set
from numpy import ndarray

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import SeqIO

from allele import AlleleList
from allele_events import AlleleEvents
from cell_state import CellState
from constants import NO_EVT_STR


class CellLineageTree(TreeNode):
    """
    History from embryo cell to observed cells. Each node represents a cell divison/death.
    Class can be used for storing information about the true cell lineage tree and can be
    used for storing the estimate of the cell lineage tree.
    """
    def __init__(self,
                 allele_list: AlleleList = None,
                 allele_events_list: List[AlleleEvents] = None,
                 cell_state: CellState = None,
                 dist: float = 0,
                 dead: bool = False,
                 abundance: int = 1,
                 resolved_multifurcation: bool = False):
        """
        @param allele OR allele_events: the allele at the CLT node.
                            Only one of these two values should be given
                            as input.
        @param cell_state: the cell state at the node
        @param dist: branch length from parent node
        @param dead: if the cell at that node is dead
        @param abundance: number of cells this node represents. Always one in this implementation
        @param resolved_multifurcation: whether this node is actually a multifurcation (as opposed to an unresolved one)
                                        Basically this is a flag for manually specifying whether or not things are resolved
        """
        super().__init__()
        self.dist = dist
        if allele_list is not None:
            self.set_allele_list(allele_list)
        elif allele_events_list is not None:
            self.add_feature("allele_events_list", allele_events_list)
            # Maybe we'll need this conversion someday. For now we leave it empty.
            self.add_feature("allele_list", None)
        else:
            raise ValueError("no alleles passed in")
        self.sync_allele_events_list_str()

        self.add_feature("cell_state", cell_state)
        self.add_feature("dead", dead)
        self.add_feature("abundance", abundance)
        self.add_feature("resolved_multifurcation", resolved_multifurcation)

    def set_allele_list(self, allele_list: AlleleList):
        self.add_feature("allele_list", allele_list)
        self.add_feature("allele_events_list", allele_list.get_event_encoding())

    def sync_allele_events_list_str(self):
        """
        Sync the string attribute with the other allele_events_list attirubte
        """
        self.add_feature("allele_events_list_str", CellLineageTree._allele_list_to_str(self.allele_events_list))

    def is_many_furcating(self):
        return len(self.get_children()) > 2

    def is_resolved_multifurcation(self):
        """
        Is this multifurcation resolved. It is resolved if it is:
          1. manually set to resolved
          2. has no more than 2 children
        """
        return len(self.get_children()) <= 2 or self.resolved_multifurcation

    def get_parsimony_score(self):
        """
        A very special function
        This only makes sense if all internal nodes are labeled with allele events!
        @return parsimony score
        """
        pars_score = 0
        for node in self.traverse("preorder"):
            if not node.is_root():
                node_evts = [set(allele_evts.events) for allele_evts in node.allele_events_list]
                node_up_evts = [set(allele_evts.events) for allele_evts in node.up.allele_events_list]
                num_evts = sum([
                    len(n_evts - n_up_evts) for n_evts, n_up_evts in zip(node_evts, node_up_evts)])
                pars_score += num_evts
        return pars_score

    def label_tree_with_strs(self):
        """
        Updates the `allele_events_list_str` attribute for all nodes in this tree
        """
        # note: tree traversal order doesn't matter
        for node in self.traverse("preorder"):
            node.allele_events_list_str = CellLineageTree._allele_list_to_str(node.allele_events_list)

    def restrict_barcodes(self, bcode_idxs: ndarray):
        """
        @param bcode_idxs: the indices of the barcodes we observe
        Update the alleles for each node in the tree to correspond to only the barcodes indicated
        """
        for node in self.traverse():
            node.allele_events_list = [node.allele_events_list[i] for i in bcode_idxs]
        self.label_tree_with_strs()

    def get_max_depth(self):
        """
        @return maximum number of nodes between leaf and root
        """
        max_depth = 0
        for leaf in self:
            node = leaf
            depth = 0
            while node.up is not None:
                node = node.up
                depth += 1
            max_depth = max(depth, max_depth)
        return max_depth

    def up_generations(self, k:int):
        """
        @return the ancestor that is `k` generations ago, stops at root
                ex: k = 0 means it returns itself
        """
        anc = self
        for i in range(k):
            if anc.is_root():
                break
            anc = anc.up
        return anc

    def label_node_ids(self, order="preorder"):
        """
        Label each node with `node_id` attribute.
        Supposes we are starting from this node, which is the root node
        Numbers nodes according to order in preorder traversal

        @return number of nodes
        """
        assert order == "preorder"
        assert self.is_root()
        node_id = 0
        for node in self.traverse(order):
            node.add_feature("node_id", node_id)
            if node.is_root():
                root_node_id = node_id
            node_id += 1
        assert root_node_id == 0
        return node_id

    def get_num_nodes(self):
        assert self.is_root()
        return len([_ for _ in self.traverse("preorder")])

    @staticmethod
    def prune_tree(clt, keep_leaf_ids: Set[int]):
        """
        prune the tree to only keep the leaves indicated
        custom pruning (cause ete was doing weird things...)
        @param clt: CellLineageTree
        @return a copy of the clt but properly pruned
        """
        # Importing here because otherwise we have circular imports... oops
        import collapsed_tree

        assert clt.is_root()
        clt_copy = clt.copy()
        for node in clt_copy.iter_descendants():
            if sum((node2.node_id in keep_leaf_ids) for node2 in node.traverse()) == 0:
                node.detach()
        collapsed_tree._remove_single_child_unobs_nodes(clt_copy)
        return clt_copy

    def copy_single(self):
        """
        @return a new CellLienageTree object but no children
        """
        copy_of_me = self.copy()
        for child in copy_of_me.get_children():
            child.detach()
        return copy_of_me

    @staticmethod
    def convert(node: TreeNode,
                 allele_list: AlleleList = None,
                 allele_events_list: List[AlleleEvents] = None,
                 cell_state: CellState = None,
                 dist: float = 0,
                 dead: bool = False,
                 abundance: int = 1,
                 resolved_multifurcation: bool = False):
        """
        Converts a TreeNode to a CellLineageTree
        @return CellLienageTree
        """
        new_node = CellLineageTree(
                 allele_list,
                 allele_events_list,
                 cell_state,
                 dist,
                 dead,
                 abundance,
                 resolved_multifurcation)
        # Don't override features that were in the new_node already.
        # Just copy over features that are missing in new_node
        for k in node.features:
            if k not in new_node.features:
                new_node.add_feature(k, getattr(node, k))
        return new_node

    @staticmethod
    def _allele_list_to_str(allele_evts_list):
        return_str = "||".join([str(a) for a in allele_evts_list])
        if return_str == "":
            return NO_EVT_STR
        else:
            return return_str

    """
    Functions for writing sequences as fastq output
    """
    def _create_sequences(self):
        """
        @return sequences for leaf alleles
        """
        sequences = []
        for i, leaf in enumerate(self, 1):
            name = 'b{}'.format(i)
            allele_sequence = re.sub('[-]', '',
                                      ''.join(leaf.allele.allele)).upper()
            indel_events = ','.join(':'.join([
                str(start), str(end), str(insertion)
            ]) for start, end, insertion in leaf.allele.get_events())
            sequences.append(
                SeqRecord(
                    Seq(allele_sequence, generic_dna),
                    id=name,
                    description=indel_events,
                    letter_annotations=dict(
                        phred_quality=[60] * len(allele_sequence))))
        return sequences

    def write_sequences(self, file_name: str):
        sequences = self._create_sequences()
        SeqIO.write(sequences, open(file_name, 'w'), 'fastq')
