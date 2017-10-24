#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import pickle
import sys
if sys.version_info < (3, 0):
    from string import maketrans
else:
    maketrans = str.maketrans
from collections import Counter
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
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO, SeqIO
from ete3 import TreeNode, NodeStyle, TreeStyle, faces, SeqMotifFace, add_face_to_node
from collapsed_tree import CollapsedTree

from constants import BARCODE_V7


class Barcode:
    '''
    GESTALT target array with spacer sequences
    v7 barcode from GESTALT paper Table S4 is unedited barcode
    initial barcode state equal to v7 by default
    '''

    def __init__(self,
                 target_lambdas=scipy.ones(10),
                 repair_lambda=10,
                 repair_indel_probability=.1,
                 repair_deletion_lambda=3,
                 repair_insertion_lambda=2,
                 barcode=BARCODE_V7,
                 unedited_barcode=BARCODE_V7):
        # validate arguments
        for i, target_lambda in enumerate(target_lambdas, 1):
            if target_lambda < 0:
                raise ValueError('{}th target rate {} is negative'.format(
                    i, target_lambda))
        if repair_lambda < 0:
            raise ValueError(
                'repair rate {} is negative'.format(repair_lambda))
        if not (0 <= repair_indel_probability <= 1):
            raise ValueError(
                'repair indel probability {} is outside the unit interval'.
                format(repair_indel_probability))
        if repair_deletion_lambda < 0:
            raise ValueError('repair deletion parameter {} is negative'.format(
                repair_deletion_lambda))
        if repair_insertion_lambda < 0:
            raise ValueError(
                'repair insertion parameter {} is negative'.format(
                    repair_insertion_lambda))
        if set(''.join(unedited_barcode)) != set('ACGT'):
            raise ValueError(
                'barcode sequence {} must contain only capital letters (inserstions will be denoted in lower case)'.
                format(unedited_barcode))
        # The original barcode
        self.unedited_barcode = unedited_barcode
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(barcode)
        # number of targets
        self.n_targets = (len(self.barcode) - 1) // 2
        if len(target_lambdas) != self.n_targets:
            raise ValueError('must give {} target_lambdas'.format(
                self.n_targets))
        # poisson rates for each target
        self.target_lambdas = scipy.array(target_lambdas)
        # poisson rate for repair process
        self.repair_lambda = repair_lambda
        # probability that repair will cause indel
        # NOTE: this is effectively a zero-inflation parameter
        self.repair_indel_probability = repair_indel_probability
        # poisson parameter for how much (~symmetric) deletion about cut sites
        # happens if deletion happens with repair
        self.repair_deletion_lambda = repair_deletion_lambda
        # poisson parameter for how much insertion at cut sites
        self.repair_insertion_lambda = repair_insertion_lambda
        # a list of target indices that have a DSB and need repair
        self.needs_repair = set()

    def indel(self, target1, target2, deletion_length=0, insertion=''):
        '''
        a utility function for deletion
        if target1 != target2, create an inter-target deletion, otherwise focal deletion
        deletion_length is symmetrically deleted about cut site(s)
        insertion sequence is placed between target deletions
        '''
        # indices into the self.barcode list (accounting for the spacers)
        index1 = 1 + 2 * min(target1, target2)
        index2 = 1 + 2 * max(target1, target2)
        # offset from 3' end of target for Cas9 cutting
        cut_site = 6  # NOTE: need to ask Aaron about this
        # sequence left of cut
        left = ','.join(self.barcode[:index1 + 1])[:-cut_site]
        # barcode sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            center = '-' * cut_site + ',' + re.sub('[acgt]', '', ','.join(
                self.barcode[(index1 + 1):index2])).translate(
                    maketrans('ACGT', '----')) + ',' + '-' * (
                        len(self.barcode[index2]) - cut_site)
        # sequence right of cut
        right = ','.join(
            self.barcode[index2:])[len(self.barcode[index2]) - cut_site:]
        # left delete
        deleted = 0
        for position, letter in reversed(list(enumerate(left))):
            if deleted == deletion_length:
                break
            if letter is not ',':
                left = left[:position] + '-' + left[(position + 1):]
                deleted += 1
        # right delete
        deleted = 0
        for position, letter in enumerate(right):
            if deleted == deletion_length:
                break
            if letter is not ',':
                right = right[:position] + '-' + right[(position + 1):]
                deleted += 1
        # put it back together
        self.barcode = (left + insertion + center + right).split(',')

    def repair(self):
        '''
        repair DSB cuts
        - if self.needs_repair is empty, do nothing
        - otherwise, create a repair/deletion, randomly choosing two indices with replacement (inter-target if two different indices)
        - deletions can span target boundaries
        - any targets that are modified get their lambda sent to zero
        '''
        if len(self.needs_repair) == 0:
            raise RuntimeError('cannot repair if not targets need repair')
        # random draw for whether or not there will be an indel
        if random() < self.repair_indel_probability:
            # a random draw for symmetric deletion about cut site(s)
            deletion_length = poisson.rvs(self.repair_deletion_lambda)
            # random insertion
            insertion_length = poisson.rvs(self.repair_insertion_lambda)
        else:
            deletion_length = 0
            insertion_length = 0
        # The inserted nucleotide content is just uniform random
        insertion = ''.join(choice(list('acgt'), insertion_length))
        # choose a random pair of targets among those that need repair (with replacement)
        # if two different targets are chose, that will result in an inter-target indel
        target1, target2 = choice(list(self.needs_repair), 2)
        # create indel
        self.indel(target1, target2, deletion_length, insertion)
        # update target_lambdas
        for target in range(self.n_targets):
            index = 1 + 2 * target
            # NOTE: this is more robust than checking for gap characters, since we might later model insertions
            if self.barcode[index] != self.unedited_barcode[index]:
                self.target_lambdas[target] = 0
        # update which targets still need repair
        self.needs_repair = {
            target
            for target in self.needs_repair
            if target < min(target1, target2)
            or target > max(target1, target2)
        }

    def simulate(self, time):
        '''
        simulate a specified time of editing
        - possible that repair event leaves no indel
        - barcode may be rendered uneditable at all targets (exhausted)
        '''
        if time <= 0:
            raise ValueError('simulation time {} is not positive'.format(time))
        t = 0
        while True:
            # Line up the target cut rates and the repair process rate, so they can race!
            # There's really a repair process at each cut site that needs repair (self.needs_repair),
            # but since repair rates are equal we can aggregate rates, and then choose one if the
            # aggregated repair event wins
            event_lambdas = scipy.concatenate([
                self.target_lambdas,
                [len(self.needs_repair) * self.repair_lambda]
            ])
            event_lambdas_total = event_lambdas.sum()
            # if barcode exhausted, we are done
            if event_lambdas_total == 0:
                break
            # add time to the next event
            t += expon.rvs(scale=1 / event_lambdas_total)
            # if we run out of time, we are done
            if t > time:
                break
            # pick an event
            event = choice(
                self.n_targets + 1, p=event_lambdas / event_lambdas_total)
            # the last event represents the aggregated repair event
            if event == self.n_targets:
                self.repair(
                )  # NOTE: random selection of which targets to repair happens in here
            else:
                self.needs_repair.add(
                    event
                )  # NOTE: if this target already needed repair, this doesn't change anything

    def events(self):
        '''return the list of observable indel events in the barcdoe'''
        events = []
        insertion_total = 0
        # find the indels
        for indel in re.compile('[-acgt]+').finditer(str(self)):
            start = indel.start() - insertion_total
            # find the insertions(s) in this indel
            insertion = ''.join(
                insertion.group(0)
                for insertion in re.compile('[acgt]+').finditer(
                    indel.group(0)))
            insertion_total = +len(insertion)
            end = indel.end() - insertion_total
            events.append((start, end, insertion))
        return events

    def __repr__(self):
        return str(''.join(self.barcode))


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
        t = expon.rvs(scale=1 / self.birth_lambda)
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
            daughter1 = BarcodeTree(
                node.barcode, birth_lambda=self.birth_lambda)
            daughter1.tree.barcode.needs_repair = set()
            # oooh, recursion
            daughter1.simulate(time - t, root=False)
            node.add_child(daughter1.tree)
            daughter2 = BarcodeTree(
                node.barcode, birth_lambda=self.birth_lambda)
            daughter2.tree.barcode.needs_repair = set()
            daughter2.simulate(time - t, root=False)
            node.add_child(daughter2.tree)

        if root:
            self.sequences = self.create_sequences(self.tree)
            self.collapsed_tree = CollapsedTree(self.tree)
            # NOTE: sequences do not necessary correspond to collapsed_tree
            #       due to possible homoplasy

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
            barcode_sequence = re.sub('[-]', '',
                                      ''.join(leaf.barcode.barcode)).upper()
            indel_events = ','.join(':'.join([
                str(start), str(end), str(insertion)
            ]) for start, end, insertion in leaf.barcode.events())
            sequences.append(
                SeqRecord(
                    Seq(barcode_sequence, generic_dna),
                    id=name,
                    description=indel_events,
                    letter_annotations=dict(
                        phred_quality=[60] * len(barcode_sequence))))
        return sequences

    def write_sequences(self, file):
        SeqIO.write(self.sequences, open(file, 'w'), 'fastq')

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
        plt.figure(figsize=(5, 1.5))
        position = 0
        # loop through and get the deletion frequency of each site
        for bit_index, bit in enumerate(self.tree.barcode.unedited_barcode):
            if len(
                    bit
            ) == 4:  # the spacer seqs are length 4, we plot vertical bars to demarcate target boundaries
                plt.bar(position, 100, 4, facecolor='black', alpha=.2)
            for bit_position, letter in enumerate(bit):
                deletion_frequency.append(100 * sum(
                    re.sub('[acgt]', '', leaf.barcode.barcode[bit_index])[
                        bit_position] == '-' for leaf in self.tree) / n_leaves)
            position += len(bit)
        plt.plot(deletion_frequency, color='red', lw=2, clip_on=False)
        # another loop through to find the frequency that each site is the start of an insertion
        insertion_start_frequency = scipy.zeros(len(str(self.tree.barcode)))
        for leaf in self.tree:
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
        plt.savefig(file)

    def indel_boundary(self, file):
        '''plot a scatter of indel start/end positions'''
        indels = pd.DataFrame(columns=('indel start', 'indel end'))
        i = 0
        for leaf in self.tree:
            for match in re.compile('[-]+').finditer(
                    re.sub('[acgt]', '', ''.join(leaf.barcode.barcode))):
                indels.loc[i] = match.start(), match.end()
                i += 1
        bc_len = len(''.join(self.tree.barcode.unedited_barcode))
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
        for bit in self.tree.barcode.unedited_barcode:
            if len(
                    bit
            ) == 4:  # the spacer seqs are length 4, we plot bars to demarcate target boundaries
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
        plt.savefig(file)

    def event_joint(self, file):
        '''make a seaborn pairgrid plot showing deletion length, 3' deltion length, and insertion length'''
        raise NotImplementedError(
            "not correctly implemented, can't identify 5' from 3' when there is no insertion"
        )
        indels = pd.DataFrame(columns=(
            "5' deletion length", "3' deletion length", 'insertion length'))
        i = 0
        for leaf in self.tree:
            for indel in re.compile(r'(-*)([acgt]*)(-*)+').finditer(
                    str(leaf.barcode)):
                if len(indel.group(0)) > 0:
                    indels.loc[i] = (len(indel.group(1)) + len(indel.group(3)),
                                     len(indel.group(2)))
                    i += 1
        plt.figure(figsize=(3, 3))
        sns.pairplot(indels)
        plt.tight_layout()
        plt.savefig(file)

    def n_leaves(self):
        return len(self.tree)


class BarcodeForest():
    '''
    simulate forest of BarcodeTree, all same parameters
    '''

    def __init__(self, barcode, birth_lambda, time=None, n=10,
                 min_leaves=None):
        self.trees = []
        ct = 0
        while len(self.trees) < n:
            tree = BarcodeTree(barcode, birth_lambda, time=time)
            if min_leaves is None or tree.n_leaves() >= min_leaves:
                self.trees.append(tree)
                ct += 1
                print('trees simulated: {} of {}  \r'.format(ct, n), end='')
        print()

    def editing_profile(self, outbase):
        '''plot profile of deletion frequency at each position over leaves'''
        for i, tree in enumerate(self.trees, 1):
            tree.editing_profile('{}.{}.editing_profile.pdf'.format(
                outbase, i))

    def indel_boundary(self, outbase):
        '''indel start/end plot for each tree'''
        for i, tree in enumerate(self.trees, 1):
            tree.indel_boundary('{}.{}.indel_boundary.pdf'.format(outbase, i))

    def event_joint(self, outbase):
        '''joint event stats'''
        for i, tree in enumerate(self.trees, 1):
            tree.event_joint('{}.{}.event_joint.pdf'.format(outbase, i))

    def summary_plots(self, file):
        n_cells = []
        n_genotypes = []
        n_indels = []
        indel_lens = []
        for tree in self.trees:
            # counter for the unique leaf genotypes
            genotypes = Counter(
                [''.join(leaf.barcode.barcode) for leaf in tree.tree])
            n_genotypes.append(len(genotypes))
            n_cells.append(sum(genotypes.values()))
            n_indels.append([
                len(re.findall('[-acgt]+', genotype))
                for genotype in genotypes.elements()
            ])
            indel_lens.append([
                len(run.group(0))
                for genotype in genotypes.elements()
                for run in re.finditer('-+', genotype)
            ])

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.hist(n_cells, stacked=True)
        plt.xlabel('number of cells')
        plt.xlim([0, None])
        plt.subplot(1, 4, 2)
        plt.hist(n_genotypes, stacked=True)
        plt.xlabel('number of genotypes')
        plt.xlim([0, None])
        plt.subplot(1, 4, 3)
        plt.hist(n_indels, stacked=True)
        # for x in n_deletions:
        #     sns.distplot(x, hist=False)
        plt.xlabel('number of indels')
        plt.xlim([0, None])
        plt.subplot(1, 4, 4)
        plt.hist(indel_lens, stacked=True)
        # for x in deletion_lens:
        #     sns.distplot(x, hist=False)
        plt.xlabel('indel lengths')
        plt.xlim([0, None])
        sns.despine()
        plt.tight_layout()
        plt.savefig(file)

    def write_sequences(self, outbase):
        for i, tree in enumerate(self.trees, 1):
            tree.write_sequences('{}.{}.fastq'.format(outbase, i))

    def render(self, outbase):
        for i, tree in enumerate(self.trees, 1):
            tree.render('{}.{}.pdf'.format(outbase, i))


def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        'outbase', type=str, help='base name for plot and fastq output')
    parser.add_argument(
        '--target_lambdas',
        type=float,
        nargs='+',
        default=[1 for _ in range(10)],
        help='target cut poisson rates')
    parser.add_argument(
        '--repair_lambda', type=float, default=10, help='repair poisson rate')
    parser.add_argument(
        '--repair_indel_probability',
        type=float,
        default=.1,
        help='probability of deletion during repair')
    parser.add_argument(
        '--repair_deletion_lambda',
        type=float,
        default=2,
        help=
        'poisson parameter for distribution of symmetric deltion about cut site(s)'
    )
    parser.add_argument(
        '--repair_insertion_lambda',
        type=float,
        default=.5,
        help='poisson parameter for distribution of insertion in cut site(s)')
    parser.add_argument(
        '--birth_lambda', type=float, default=1, help='birth rate')
    parser.add_argument(
        '--time', type=float, default=5, help='how much time to simulate')
    parser.add_argument(
        '--min_leaves',
        type=int,
        default=0,
        help='condition on at least this many leaves')
    parser.add_argument(
        '--n_trees', type=int, default=1, help='number of trees in forest')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    forest = BarcodeForest(
        Barcode(
            target_lambdas=args.target_lambdas,
            repair_lambda=args.repair_lambda,
            repair_indel_probability=args.repair_indel_probability,
            repair_deletion_lambda=args.repair_deletion_lambda,
            repair_insertion_lambda=args.repair_insertion_lambda),
        birth_lambda=args.birth_lambda,
        time=args.time,
        min_leaves=args.min_leaves,
        n=args.n_trees)
    forest.editing_profile(args.outbase)
    forest.indel_boundary(args.outbase)
    # NOTE: function below not yet implemented
    # forest.event_joint(args.outbase)
    forest.write_sequences(args.outbase)
    forest.render(args.outbase)
    forest.summary_plots(args.outbase + '.summary_plots.pdf')

    with open(args.outbase + ".pkl", "wb") as f_pkl:
        pickle.dump(forest, f_pkl)


if __name__ == "__main__":
    main()
