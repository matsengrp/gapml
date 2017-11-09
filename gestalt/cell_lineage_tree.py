import re
import scipy
import pandas as pd
from ete3 import TreeNode, NodeStyle, SeqMotifFace, TreeStyle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_style('ticks')

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO, SeqIO

from barcode import Barcode
from barcode_events import BarcodeEvents
from cell_state import CellState
from common import get_color


class CellLineageTree(TreeNode):
    """
    History from embryo cell to observed cells. Each node represents a cell divison/death.
    Class can be used for storing information about the true cell lineage tree and can be
    used for storing the estimate of the cell lineage tree.
    """

    def __init__(self,
                 barcode: Barcode = None,
                 barcode_events: BarcodeEvents = None,
                 cell_state: CellState = None,
                 dist: float = 0,
                 dead: bool = False,
                 n_id: int = None):
        """
        @param barcode: the barcode at the CLT node -- this is allowed to be None
        @param cell_state: the cell state at the node
        @param dist: branch length from parent node
        @param dead: if the cell at that node is dead
        @param n_id: a node id -- useful for estimation
        """
        super().__init__()
        self.dist = dist
        if barcode is not None:
            self.add_feature("barcode", barcode)
            self.add_feature("barcode_events", barcode.get_event_encoding())
        else:
            self.add_feature("barcode_events", barcode_events)
            self.add_feature("barcode", None)

        self.add_feature("cell_state", cell_state)
        self.add_feature("dead", dead)
        self.add_feature("id", n_id)

    def _create_sequences(self):
        """
        @return sequences for leaf barcodes
        """
        sequences = []
        for i, leaf in enumerate(self, 1):
            name = 'b{}'.format(i)
            barcode_sequence = re.sub('[-]', '',
                                      ''.join(leaf.barcode.barcode)).upper()
            indel_events = ','.join(':'.join([
                str(start), str(end), str(insertion)
            ]) for start, end, insertion in leaf.barcode.get_events())
            sequences.append(
                SeqRecord(
                    Seq(barcode_sequence, generic_dna),
                    id=name,
                    description=indel_events,
                    letter_annotations=dict(
                        phred_quality=[60] * len(barcode_sequence))))
        return sequences

    def write_sequences(self, file_name: str):
        sequences = self._create_sequences()
        SeqIO.write(sequences, open(file_name, 'w'), 'fastq')

    def savefig(self, file_name: str):
        '''render tree to image file_name'''
        for n in self.traverse():
            style = NodeStyle()
            style['size'] = 5
            style['fgcolor'] = get_color(n.cell_state.categorical_state.cell_type)
            n.set_style(style)
        for leaf in self:
            # get the motif list for indels in the format that SeqMotifFace expects
            motifs = []
            for match in re.compile('[acgt]+').finditer(str(leaf.barcode)):
                motifs.append([
                    match.start(),
                    match.end(), '[]',
                    match.end() - match.start(), 1, 'blue', 'blue', None
                ])
            seqFace = SeqMotifFace(
                seq=str(leaf.barcode).upper(),
                motifs=motifs,
                seqtype='nt',
                seq_format='[]',
                height=3,
                gapcolor='red',
                gap_format='[]',
                fgcolor='black',
                bgcolor='lightgrey',
                width=5)
            leaf.add_face(seqFace, 0, position="aligned")
        tree_style = TreeStyle()
        tree_style.show_scale = False
        tree_style.show_leaf_name = False
        self.render(file_name, tree_style=tree_style)

    def editing_profile(self, file_name: str):
        '''plot profile_name of deletion frequency at each position over leaves'''
        n_leaves = len(self)
        deletion_frequency = []
        plt.figure(figsize=(5, 1.5))
        position = 0
        # loop through and get the deletion frequency of each site
        for bit_index, bit in enumerate(self.barcode.unedited_barcode):
            if len(bit) == 4:
                # the spacer seqs are length 4, we plot vertical bars to demarcate target boundaries
                plt.bar(position, 100, 4, facecolor='black', alpha=.2)
            for bit_position, letter in enumerate(bit):
                deletion_frequency.append(100 * sum(
                    re.sub('[acgt]', '', leaf.barcode.barcode[bit_index])[
                        bit_position] == '-' for leaf in self) / n_leaves)
            position += len(bit)
        plt.plot(deletion_frequency, color='red', lw=2, clip_on=False)
        # another loop through to find the frequency that each site is the start of an insertion
        insertion_start_frequency = scipy.zeros(len(str(self.barcode)))
        for leaf in self:
            insertion_total = 0
            for insertion in re.compile('[acgt]+').finditer(str(leaf.barcode)):
                start = insertion.start() - insertion_total
                # find the insertions(s) in this indel
                insertion_total = +len(insertion.group(0))
                insertion_start_frequency[start] += 100 / n_leaves
        plt.plot(insertion_start_frequency, color='blue', lw=2, clip_on=False)
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
        plt.savefig(file_name)

    def indel_boundary(self, file_name: str):
        '''plot a scatter of indel start/end positions'''
        indels = pd.DataFrame(columns=('indel start', 'indel end'))
        i = 0
        for leaf in self:
            for match in re.compile('[-]+').finditer(
                    re.sub('[acgt]', '', ''.join(leaf.barcode.barcode))):
                indels.loc[i] = match.start(), match.end()
                i += 1
        bc_len = len(''.join(self.barcode.unedited_barcode))
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
        for bit in self.barcode.unedited_barcode:
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
                    str(leaf.barcode)):
                if len(indel.group(0)) > 0:
                    indels.loc[i] = (len(indel.group(1)) + len(indel.group(3)),
                                     len(indel.group(2)))
                    i += 1
        plt.figure(figsize=(3, 3))
        sns.pairplot(indels)
        plt.tight_layout()
        plt.savefig(file_name)
