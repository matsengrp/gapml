#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from string import maketrans

import scipy, argparse
from scipy.stats import expon, poisson
from numpy.random import choice, random
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import MultipleSeqAlignment
from ete3 import TreeNode, NodeStyle, TreeStyle, faces, SeqMotifFace, add_face_to_node

class Barcode:
    '''
    GESTALT target array with spacer sequences
    v7 barcode from GESTALT paper Table S4 is unedited barcode
    initial barcode state equal to v7 by default
    '''
    v7 = (  'cg', 'GATACGATACGCGCACGCTATGG',
          'agtc', 'GACACGACTCGCGCATACGATGG',
          'agtc', 'GATAGTATGCGTATACGCTATGG',
          'agtc', 'GATATGCATAGCGCATGCTATGG',
          'agtc', 'GAGTCGAGACGCTGACGATATGG',
          'agtc', 'GCTACGATACACTCTGACTATGG',
          'agtc', 'GCGACTGTACGCACACGCGATGG',
          'agtc', 'GATACGTAGCACGCAGACTATGG',
          'agtc', 'GACACAGTACTCTCACTCTATGG',
          'agtc', 'GATATGAGACTCGCATGTGATGG',
          'ga')
    def __init__(self,
                 target_lambdas=scipy.ones(10),
                 repair_lambda=10,
                 repair_deletion_probability=.1,
                 repair_deletion_lambda=2,
                 barcode=v7):
        # validate arguments
        for i, target_lambda in enumerate(target_lambdas, 1):
            if target_lambda < 0:
                raise ValueError('{}th target rate {} is negative'.format(i, target_lambda))
        if repair_lambda < 0:
            raise ValueError('repair rate {} is negative'.format(repair_lambda))
        if not (0 <= repair_deletion_probability <= 1):
            raise ValueError('repair deletion probability {} is outside the unit interval'.format(repair_deletion_probability))
        if repair_deletion_lambda < 0:
            raise ValueError('repair deletion parameter {} is negative'.format(repair_deletion_lambda))
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(barcode)
        # number of targets
        self.n_targets = (len(self.barcode) - 1)//2
        if len(target_lambdas) != self.n_targets:
            raise ValueError('must give {} target_lambdas'.format(self.n_targets))
        # poisson rates for each target
        self.target_lambdas = scipy.array(target_lambdas)
        # poisson rate for repair process
        self.repair_lambda = repair_lambda
        # probability that repair will cause deletion
        # NOTE: this is effectively a zero-inflation parameter
        self.repair_deletion_probability = repair_deletion_probability
        # poisson parameter for how much (~symmetric) deletion about cut sites
        # happens if deletion happens with repair
        self.repair_deletion_lambda = repair_deletion_lambda
        # a list of target indices that have a DSB and need repair
        self.needs_repair = set()

    def delete(self, target1, target2, deletion_length):
        '''
        a utility function for deletion
        if target1 != target2, create an inter-target deletion, otherwise focal deletion
        deletion_length is symmetrically deleted about cut site(s)
        '''
        # indices into the self.barcode list (accounting for the spacers)
        index1 = 1 + 2*min(target1, target2)
        index2 = 1 + 2*max(target1, target2)
        # offset from 3' end of target for Cas9 cutting
        cut_site = 6 # NOTE: need to ask Aaron about this
        # sequence left of cut
        left = ','.join(self.barcode[:index1 + 1])[:-cut_site]
        # barcode sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            center = '-' * cut_site + ',' + ','.join(self.barcode[(index1 + 1):index2]).translate(maketrans('ACGTacgt', '-'*8)) + ',' + '-' * (len(self.barcode[index2]) - cut_site)
        # sequence right of cut
        right = ','.join(self.barcode[index2:])[len(self.barcode[index2]) - cut_site:]
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
        self.barcode = (left + center + right).split(',')
        # sanity check
        assert len(''.join(self.barcode)) == len(''.join(self.v7))

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
        # random draw for whether or not there will be a deletion
        if random() < self.repair_deletion_probability:
            # a random draw for symmetric deletion about cut site(s)
            deletion_length = poisson.rvs(self.repair_deletion_lambda)
        else:
            deletion_length = 0
        # choose a random pair of targets among those that need repair (with replacement)
        # if two different targets are chose, that will result in an inter-target deletion
        target1, target2 = choice(list(self.needs_repair), 2)
        # create deletion
        self.delete(target1, target2, deletion_length)
        # update target_lambdas
        for target in range(self.n_targets):
            index = 1 + 2*target
            # NOTE: this is more robust than checking for gap characters, since we might later model insertions
            if self.barcode[index] != self.v7[index]:
                self.target_lambdas[target] = 0
        # update which targets still need repair
        self.needs_repair = {target for target in self.needs_repair if target < min(target1, target2) or target > max(target1, target2)}

    def simulate(self, time):
        '''
        simulate a specified time of editing
        - possible that repair event leaves no deletion
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
            event_lambdas = scipy.concatenate([self.target_lambdas, [len(self.needs_repair)*self.repair_lambda]])
            event_lambdas_total = event_lambdas.sum()
            # if barcode exhausted, we are done
            if event_lambdas_total == 0:
                break
            # add time to the next event
            t += expon.rvs(scale=1/event_lambdas_total)
            # if we run out of time, we are done
            if t > time:
                break
            # pick an event
            event = choice(self.n_targets + 1, p=event_lambdas/event_lambdas_total)
            # the last event represents the aggregated repair event
            if event == self.n_targets:
                self.repair() # NOTE: random selection of which targets to repair happens in here
            else:
                self.needs_repair.add(event) # NOTE: if this target already needed repair, this doesn't change anything

    def __repr__(self):
        return str(''.join(self.barcode))


class BarcodeTree():
    '''
    simulate tree of barcodes
    initialized with an instance of type Barcode (or any type with a simulation method)
    '''
    def __init__(self, barcode, birth_lambda=1):
        if birth_lambda < 0:
            raise ValueError('birth rate {} is negative'.format(birth_lambda))
        self.birth_lambda = birth_lambda
        self.tree = TreeNode(dist=0)
        self.tree.add_feature('barcode', barcode)

    def simulate(self, simulation_time, root=True):
        if simulation_time <= 0:
            raise ValueError('simulation time {} is not positive'.format(simulation_time))
        # time to the next division or end of simulation
        t = min(expon.rvs(scale=1/self.birth_lambda), simulation_time)

        # add child and edit the barcode for that long
        child = self.tree.copy()
        child.barcode.simulate(t)
        child.dist = t

        if t < simulation_time:
            # not a leaf, so add two daughters
            daughter1 = BarcodeTree(child.barcode)
            daughter1.simulate(simulation_time - t, root=False)
            child.add_child(daughter1.tree)
            daughter2 = BarcodeTree(child.barcode)
            daughter2.simulate(simulation_time - t, root=False)
            child.add_child(daughter2.tree)

        if root:
            self.tree.add_child(child)
            # sequence alignment for leaf barcodes
            self.aln = MultipleSeqAlignment([])
            for i, leaf in enumerate(self.tree, 1):
                name = 'barcode{}'.format(i)
                leaf.name = name
                self.aln.append(SeqRecord(Seq(str(leaf.barcode).upper(), generic_dna), id=name, description=''))
        else:
            self.tree = child

        return self.tree

    def get_unedited_barcode(self):
        return str(self.tree.barcode)

    def write_alignment(self, file):
        AlignIO.write(self.aln, open(file, 'w'), 'fasta')

    def render(self, file):
        '''render tree to image file'''
        style = NodeStyle()
        style['size'] = 0
        for n in self.tree.traverse():
           n.set_style(style)
        for leaf in self.tree:
           seqFace = SeqMotifFace(seq=str(leaf.barcode).upper(),
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
        # plot editing percentage on top of alignment
        dat = []
        n_leaves = len(self.tree)
        plt.figure(figsize=(5,1))
        for position, letter in enumerate(str(''.join(Barcode.v7))):
            if letter.islower():
                plt.bar(position, 100, 1, facecolor='black', alpha=.2)
            dat.append(100*sum(str(leaf.barcode)[position] == '-' for leaf in self.tree)/n_leaves)
        plt.plot(dat, color='red', lw=2, clip_on=False)
        plt.xlim(0, len(dat))
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


def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument('outbase', type=str, help='base name for plot and fasta output')
    parser.add_argument('--target_lambdas', type=float, nargs='+', default=[2**-n for n in range(10)], help='target cut poisson rates')
    parser.add_argument('--repair_lambda', type=float, default=10, help='repair poisson rate')
    parser.add_argument('--repair_deletion_probability', type=float, default=.1, help='probability of deletion during repair')
    parser.add_argument('--repair_deletion_lambda', type=float, default=5., help='poisson parameter for distribution of symmetric deltion about cut site(s) if deletion happens during repair')
    parser.add_argument('--birth_lambda', type=float, default=1, help='birth rate')
    parser.add_argument('--time', type=int, default=5, help='how much time to simulate')
    args = parser.parse_args()

    scipy.seterr(all='raise')

    tree = BarcodeTree(Barcode(target_lambdas=args.target_lambdas,
                               repair_lambda=args.repair_lambda,
                               repair_deletion_probability=args.repair_deletion_probability,
                               repair_deletion_lambda=args.repair_deletion_lambda))
    tree.simulate(args.time)
    tree.write_alignment(args.outbase + '.fasta')
    tree.render(args.outbase + '.tree.pdf')
    tree.editing_profile(args.outbase + '.editing_profile.pdf')

if __name__ == "__main__":
    main()
