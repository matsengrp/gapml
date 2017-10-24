import numpy as np
import scipy, argparse, copy, re
from scipy.stats import expon, poisson
from numpy.random import choice, random
import pandas as pd

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_style('ticks')

from Bio import AlignIO, SeqIO
from ete3 import TreeNode, NodeStyle, TreeStyle, faces, SeqMotifFace, add_face_to_node

class BarcodeTree():
    '''
    simulate tree of barcodes
    initialized with an instance of type Barcode (or any type with a simulation method)
    '''
    def __init__(self, barcode, birth_lambda, time=None, N=None):
        if birth_lambda < 0:
            raise ValueError('birth rate {} is negative'.format(birth_lambda))
        self.initial_barcode = barcode.barcode
        self.birth_lambda = birth_lambda
        self.tree = TreeNode(dist=0)
        self.tree.add_feature('barcode', copy.deepcopy(barcode))
        if time is not None or N is not None:
            self.simulate(time=time, N=N)

    def simulate(self, time=None, N=None, root=True):
        if time is not None and time <= 0:
            raise ValueError('simulation time {} is not positive'.format(time))
        if N is not None and not (isinstance(N, int) and N > 0):
            raise ValueError('N = {} is not a positive integer'.format(N))
        if time is None and N is None:
            raise ValueError('time and N cannot both be None')
        # NOTE: N is not implemented, need to refactor as level-order simulation
        # time to the next division or end of simulation
        t = expon.rvs(scale=1/self.birth_lambda)
        if time is not None and time < t:
            t = time
        # define node for editing, and edit its barcode for the specified time
        if root:
            # keep the unedited state above
            node = self.tree.copy()
            self.tree.add_child(node)
        else:
            node = self.tree
        node.barcode.simulate(t)
        node.dist = t

        if t < time:
            # not a leaf, so add two daughters
            # NOTE: daughters do not inherit DSBs (do not need repair)
            daughter1 = BarcodeTree(node.barcode, birth_lambda=self.birth_lambda)
            daughter1.tree.barcode.needs_repair = set()
            # oooh, recursion
            daughter1.simulate(time - t, root=False)
            node.add_child(daughter1.tree)
            daughter2 = BarcodeTree(node.barcode, birth_lambda=self.birth_lambda)
            daughter2.tree.barcode.needs_repair = set()
            daughter2.simulate(time - t, root=False)
            node.add_child(daughter2.tree)

        if root:
            self.sequences = self.create_sequences(self.tree)
            self.collapsed_tree = CollapsedTree(self.tree)
            self.collapsed_sequences = self.create_sequences(self.collapsed_tree.tree)

    @staticmethod
    def create_sequences(tree):
        """
        sequences for leaf barcodes
        warning: this will rename the leaf nodes!
        """
        sequences = []
        for i, leaf in enumerate(tree, 1):
            name = 'b{}'.format(i)
            leaf.name = name
            barcode_sequence = re.sub('[-]', '', ''.join(leaf.barcode.barcode)).upper()
            sequences.append(SeqRecord(Seq(barcode_sequence,
                                       generic_dna),
                             id=name,
                             description=','.join(':'.join([str(start), str(end), str(insertion)]) for start, end, insertion in leaf.barcode.events()),
                             letter_annotations=dict(phred_quality=[60]*len(barcode_sequence))
                             ))
        return sequences

    def write_sequences(self, file):
        SeqIO.write(self.sequences, open(file, 'w'), 'fastq')

    def write_collapsed_sequences(self, file):
        SeqIO.write(self.collapsed_sequences, open(file, 'w'), 'fastq')

    def render(self, file):
        '''render tree to image file'''
        style = NodeStyle()
        style['size'] = 0
        for n in self.tree.traverse():
           n.set_style(style)
        for leaf in self.tree:
           # get the motif list for indels in the format that SeqMotifFace expects
           motifs = []
           for match in re.compile('[acgt]+').finditer(str(leaf.barcode)):
               motifs.append([match.start(), match.end(), '[]', match.end() - match.start(), 1, 'blue', 'blue', None])
           seqFace = SeqMotifFace(seq=str(leaf.barcode).upper(),
                                  motifs=motifs,
                                  seqtype='nt',
                                  seq_format='[]',
                                  height=3,
                                  gapcolor='red',
                                  fgcolor='black',
                                  bgcolor='lightgray',
                                  width=5)
           leaf.add_face(seqFace, 1)
        tree_style = TreeStyle()
        tree_style.show_scale = False
        tree_style.show_leaf_name = False
        self.tree.render(file, tree_style=tree_style)

    def editing_profile(self, file):
        '''plot profile of deletion frequency at each position over leaves'''
        n_leaves = len(self.tree)
        deletion_frequency = []
        plt.figure(figsize=(5,1.5))
        position = 0
        # loop through and get the deletion frequency of each site
        for bit_index, bit in enumerate(self.tree.barcode.unedited_barcode):
            if len(bit) == 4: # the spacer seqs are length 4, we plot vertical bars to demarcate target boundaries
                plt.bar(position, 100, 4, facecolor='black', alpha=.2)
            for bit_position, letter in enumerate(bit):
                deletion_frequency.append(100*sum(re.sub('[acgt]', '', leaf.barcode.barcode[bit_index])[bit_position] == '-' for leaf in self.tree)/n_leaves)
            position += len(bit)
        plt.plot(deletion_frequency, color='red', lw=2, clip_on=False)
        # another loop through to find the frequency that each site is the start of an insertion
        insertion_start_frequency = scipy.zeros(len(str(self.tree.barcode)))
        for leaf in self.tree:
            insertion_total = 0
            for insertion in re.compile('[acgt]+').finditer(str(leaf.barcode)):
                start = insertion.start() - insertion_total
                # find the insertions(s) in this indel
                insertion_total =+ len(insertion.group(0))
                insertion_start_frequency[start] += 100/n_leaves
        plt.plot(insertion_start_frequency, color='blue', lw=2, clip_on=False)
        plt.xlim(0, len(deletion_frequency))
        plt.ylim(0, 100)
        plt.ylabel('Editing (%)')
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
        plt.tight_layout()
        plt.savefig(file)

    def indel_boundary(self, file):
        '''plot a scatter of indel start/end positions'''
        indels = pd.DataFrame(columns=('indel start', 'indel end'))
        i = 0
        for leaf in self.tree:
            for match in re.compile('[-]+').finditer(re.sub('[acgt]', '', ''.join(leaf.barcode.barcode))):
                indels.loc[i] = match.start(), match.end()
                i += 1
        bc_len = len(''.join(self.tree.barcode.unedited_barcode))
        plt.figure(figsize=(3, 3))
        bins = scipy.linspace(0, bc_len, 10 + 1)
        g = (sns.jointplot('indel start', 'indel end', data=indels,
                           stat_func=None, xlim=(0, bc_len - 1), ylim=(0, bc_len - 1),
                           space=0, marginal_kws=dict(bins=bins, color='gray'), joint_kws=dict(alpha=.2, marker='+', color='black', zorder=2))
             .plot_joint(plt.hist2d, bins=bins, norm=LogNorm(), cmap='Reds', zorder=0))
        position = 0
        for bit in self.tree.barcode.unedited_barcode:
            if len(bit) == 4: # the spacer seqs are length 4, we plot bars to demarcate target boundaries
                for ax in g.ax_marg_x, g.ax_joint:
                    ax.bar(position, bc_len if ax == g.ax_joint else 1, 4, facecolor='gray', lw=0, zorder=1)
                for ax in g.ax_marg_y, g.ax_joint:
                    ax.barh(position, bc_len if ax == g.ax_joint else 1, 4, facecolor='gray', lw=0, zorder=1)
            position += len(bit)
        g.ax_joint.plot([0, bc_len - 1], [0, bc_len - 1], ls='--', color='black', lw=1, alpha=.2)
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        # plt.tight_layout()
        plt.savefig(file)

    def event_joint(self, file):
        '''make a seaborn pairgrid plot showing deletion length, 3' deltion length, and insertion length'''
        raise NotImplementedError("not correctly implemented, can't identify 5' from 3' when there is no insertion")
        indels = pd.DataFrame(columns=("5' deletion length", "3' deletion length", 'insertion length'))
        i = 0
        for leaf in self.tree:
            for indel in re.compile(r'(-*)([acgt]*)(-*)+').finditer(str(leaf.barcode)):
                if len(indel.group(0)) > 0:
                    indels.loc[i] = (len(indel.group(1)) + len(indel.group(3)), len(indel.group(2)))
                    i += 1
        plt.figure(figsize=(3, 3))
        sns.pairplot(indels)
        plt.tight_layout()
        plt.savefig(file)


    def n_leaves(self):
        return len(self.tree)
