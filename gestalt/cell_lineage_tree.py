import re
import scipy
import pandas as pd
from ete3 import TreeNode #, NodeStyle, SeqMotifFace, TreeStyle, TextFace, RectFace
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_style('ticks')
from typing import List

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO, SeqIO

from allele import Allele, AlleleList
from allele_events import AlleleEvents
from cell_state import CellState
from common import get_color
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
                 n_id: int = None,
                 abundance: int = 1):
        """
        @param allele OR allele_events: the allele at the CLT node.
                            Only one of these two values should be given
                            as input.
        @param cell_state: the cell state at the node
        @param dist: branch length from parent node
        @param dead: if the cell at that node is dead
        @param n_id: a node id -- useful for estimation
        """
        super().__init__()
        self.dist = dist
        if allele_list is not None:
            self.add_feature("allele_list", allele_list)
            self.add_feature("allele_events_list", [
                allele.get_event_encoding() for allele in allele_list.alleles])
            self.add_feature("allele_events_list_str", CellLineageTree._allele_list_to_str(self.allele_events_list))
        else:
            self.add_feature("allele_events_list", allele_events_list)
            self.add_feature("allele_events_list_str", CellLineageTree._allele_list_to_str(allele_events_list))
            # Maybe we'll need this conversion someday. For now we leave it empty.
            self.add_feature("allele_list", None)

        self.add_feature("cell_state", cell_state)
        self.add_feature("dead", dead)
        self.add_feature("id", n_id)
        self.add_feature("abundance", abundance)

    def label_tree_with_strs(self):
        for node in self.traverse("preorder"):
            node.allele_events_list_str = CellLineageTree._allele_list_to_str(node.allele_events_list)

    @staticmethod
    def _allele_list_to_str(allele_evts_list):
        return_str = "||".join([str(a) for a in allele_evts_list])
        if return_str == "":
            return NO_EVT_STR
        else:
            return return_str

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

    def savefig(self, file_name: str):
        '''render tree to image file_name'''
        # we make a copy, so face attributes are not retained in self
        self_copy = self.copy()
        max_abundance = max(leaf.abundance for leaf in self_copy)
        for n in self_copy.traverse():
            style = NodeStyle()
            style['size'] = 5
            style['fgcolor'] = 'black' if n.cell_state is None else get_color(n.cell_state.categorical_state.cell_type)
            n.set_style(style)
        for leaf in self_copy:
            # get the motif list for indels in the format that SeqMotifFace expects
            motifs = []
            for match in re.compile('[acgt]+').finditer(str(leaf.allele)):
                motifs.append([
                    match.start(),
                    match.end(), '[]',
                    match.end() - match.start(), 10, 'black', 'blue', None
                ])
            for match in re.compile('[-]+').finditer(str(leaf.allele)):
                motifs.append([
                    match.start(),
                    match.end(), '[]',
                    match.end() - match.start(), 10, 'black', 'red', None
                ])
            seqFace = SeqMotifFace(
                seq=str(leaf.allele).upper(),
                motifs=motifs,
                seqtype='nt',
                seq_format='[]',
                height=10,
                gapcolor='red',
                gap_format='[]',
                fgcolor='black',
                bgcolor='lightgrey')
            leaf.add_face(seqFace, 0, position="aligned")
            T = TextFace(text=leaf.abundance)
            if max_abundance > 1:
                leaf.add_face(T, 1, position="aligned")
                R = RectFace(100*leaf.abundance/max_abundance, 10, 'black', 'black')
                leaf.add_face(R, 2, position="aligned")
        tree_style = TreeStyle()
        tree_style.show_scale = False
        tree_style.show_leaf_name = False
        # NOTE: need to return for display in IPython using file_name = "%%inline"
        return self_copy.render(file_name, tree_style=tree_style)

    def editing_profile(self, file_name: str = None):
        '''
        plot profile_name of deletion frequency at each position over leaves
        @param file_name: name of file to save, None for no savefig
        @return: figure handle
        '''
        n_leaves = sum(leaf.abundance for leaf in self)
        deletion_frequency = []
        fig = plt.figure(figsize=(5, 1.5))
        position = 0
        # loop through and get the deletion frequency of each site
        for bit_index, bit in enumerate(self.allele.unedited_allele):
            if len(bit) == 4:
                # the spacer seqs are length 4, we plot vertical bars to demarcate target boundaries
                plt.bar(position, 100, 4, facecolor='black', alpha=.2)
            for bit_position, letter in enumerate(bit):
                deletion_frequency.append(100 * sum(
                    leaf.abundance * int(re.sub('[acgt]', '', leaf.allele.allele[bit_index])[
                        bit_position] == '-') for leaf in self) / n_leaves)
            position += len(bit)
        plt.plot(deletion_frequency, color='red', lw=2, clip_on=False)
        # another loop through to find the frequency that each site is the start of an insertion
        insertion_flank_frequency = scipy.zeros(len(str(self.allele)))
        for leaf in self:
            insertion_total = 0
            for insertion in re.compile('[acgt]+').finditer(str(leaf.allele)):
                start = insertion.start() - insertion_total
                end = insertion.end() - insertion_total
                insertion_flank_frequency[start:end] += 100 * leaf.abundance / n_leaves
                insertion_total += len(insertion.group(0))
        plt.plot(insertion_flank_frequency, color='blue', lw=2, clip_on=False)
        plt.xlim(0, len(deletion_frequency))
        plt.ylim(0, 100)
        plt.ylabel('Editing (%)')
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name)
        return fig

    def indel_boundary(self, file_name: str):
        '''plot a scatter of indel start/end positions'''
        indels = pd.DataFrame(columns=('indel start', 'indel end'))
        i = 0
        for leaf in self:
            for match in re.compile('[-]+').finditer(
                    re.sub('[acgt]', '', ''.join(leaf.allele.allele))):
                indels.loc[i] = match.start(), match.end()
                i += 1
        bc_len = len(''.join(self.allele.unedited_allele))
        plt.figure(figsize=(3, 3))
        bins = scipy.linspace(0, bc_len, 10 + 1)
        g = (sns.jointplot(
            'indel start',
            'indel end',
            data=indels,
            stat_func=None,
            xlim=(0, bc_len - 1),
            ylim=(0, bc_len - 1),
            space=0,
            marginal_kws=dict(bins=bins, color='gray'),
            joint_kws=dict(alpha=.2, marker='+', color='black', zorder=2))
             .plot_joint(
                 plt.hist2d, bins=bins, norm=LogNorm(), cmap='Reds', zorder=0))
        position = 0
        for bit in self.allele.unedited_allele:
            if len(bit) == 4:
                # the spacer seqs are length 4, we plot bars to demarcate target boundaries
                for ax in g.ax_marg_x, g.ax_joint:
                    ax.bar(
                        position,
                        bc_len if ax == g.ax_joint else 1,
                        4,
                        facecolor='gray',
                        lw=0,
                        zorder=1)
                for ax in g.ax_marg_y, g.ax_joint:
                    ax.barh(
                        position,
                        bc_len if ax == g.ax_joint else 1,
                        4,
                        facecolor='gray',
                        lw=0,
                        zorder=1)
            position += len(bit)
        g.ax_joint.plot(
            [0, bc_len - 1], [0, bc_len - 1],
            ls='--',
            color='black',
            lw=1,
            alpha=.2)
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        # plt.tight_layout()
        plt.savefig(file_name)

    def event_joint(self, file_name: str):
        '''make a seaborn pairgrid plot showing deletion length, 3' deltion length, and insertion length'''
        raise NotImplementedError(
            "not correctly implemented, can't identify 5' from 3' when there is no insertion"
        )
        indels = pd.DataFrame(columns=(
            "5' deletion length", "3' deletion length", 'insertion length'))
        i = 0
        for leaf in self:
            for indel in re.compile(r'(-*)([acgt]*)(-*)+').finditer(
                    str(leaf.allele)):
                if len(indel.group(0)) > 0:
                    indels.loc[i] = (len(indel.group(1)) + len(indel.group(3)),
                                     len(indel.group(2)))
                    i += 1
        plt.figure(figsize=(3, 3))
        sns.pairplot(indels)
        plt.tight_layout()
        plt.savefig(file_name)

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

    @staticmethod
    def convert(node: TreeNode,
                 allele_list: AlleleList = None,
                 allele_events_list: List[AlleleEvents] = None,
                 cell_state: CellState = None,
                 dist: float = 0,
                 dead: bool = False,
                 n_id: int = None,
                 abundance: int = 1):
        new_node = CellLineageTree(
                 allele_list,
                 allele_events_list,
                 cell_state,
                 dist,
                 dead,
                 n_id,
                 abundance)
        for k in node.features:
            if k not in new_node.features:
                new_node.add_feature(k, getattr(node, k))
        return new_node

    #def get_robinson_foulds_collapsed(self, coll_tree2, attr1, attr2, idxs=None):
    #    def make_fake_leaves(tree, attr):
    #        for node in tree.traverse():
    #            if getattr(node, attr) and not node.is_leaf():
    #                new_child = CellLineageTree.convert(
    #                        node,
    #                        node.allele_list,
    #                        node.allele_events_list,
    #                        node.cell_state,
    #                        0)
    #                node.add_child(new_child)

    #    tree1 = self.copy()
    #    tree2 = coll_tree2.copy()
    #    if idxs is not None:
    #        tree1 = CollapsedTree.collapse_same_ancestral(tree1, idxs=idxs)
    #        tree2 = CollapsedTree.collapse_same_ancestral(tree2, idxs=idxs)

    #    make_fake_leaves(tree1, attr1)
    #    make_fake_leaves(tree2, attr2)
    #    rf_dist_res = tree1.robinson_foulds(
    #            tree2,
    #            attr_t1=attr1,
    #            attr_t2=attr2,
    #            # The expand_polytomies option is buggy.
    #            # Also, I think we actually want the unrooted calculation.
    #            expand_polytomies=False,
    #            unrooted_trees=True)
    #    return rf_dist_res[0], rf_dist_res[1]

    def get_root_to_observed_lens(self, est_branch_lens = None):
        use_dist = est_branch_lens is None
        est_dist_to_root = {}
        stupid_dist_to_root = {}
        for node in self.traverse():
            if node.observed:
                est_dist = node.dist if use_dist else est_branch_lens[node.node_id]
                stupid_dist = 1
                curr_node = node
                while not curr_node.up.is_root():
                    est_dist += curr_node.up.dist if use_dist else est_branch_lens[curr_node.up.node_id]
                    stupid_dist += 1
                    curr_node = curr_node.up
                est_dist_to_root[node.allele_events_list_str] = est_dist
                stupid_dist_to_root[node.allele_events_list_str] = stupid_dist
        return est_dist_to_root, stupid_dist_to_root

    def label_node_ids(self, order):
        node_id = 0
        for node in self.traverse(order):
            node.add_feature("node_id", node_id)
            if node.is_root():
                root_node_id = node_id
            node_id += 1
        return root_node_id, node_id
