#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from collections import Counter
import scipy, argparse, copy, re
from scipy.stats import expon, poisson
from numpy.random import choice, random
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_style('ticks')
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
            center = '-' * cut_site + ',' + ','.join(self.barcode[(index1 + 1):index2]).translate(str.maketrans('ACGTacgt', '-'*8)) + ',' + '-' * (len(self.barcode[index2]) - cut_site)
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
    def __init__(self, barcode, birth_lambda, simulation_time=None):
        if birth_lambda < 0:
            raise ValueError('birth rate {} is negative'.format(birth_lambda))
        self.birth_lambda = birth_lambda
        self.tree = TreeNode(dist=0)
        self.tree.add_feature('barcode', copy.deepcopy(barcode))
        if simulation_time is not None:
            self.simulate(simulation_time)

    def simulate(self, simulation_time, root=True):
        if simulation_time <= 0:
            raise ValueError('simulation time {} is not positive'.format(simulation_time))
        # time to the next division or end of simulation
        t = min(expon.rvs(scale=1/self.birth_lambda), simulation_time)
        # define node for editing, and edit its barcode for the specified time
        if root:
            # keep the unedited state above
            node = self.tree.copy()
            self.tree.add_child(node)
        else:
            node = self.tree
        node.barcode.simulate(t)
        node.dist = t

        if t < simulation_time:
            # not a leaf, so add two daughters
            # daughters do not inherit DSBs (do not need repair)
            daughter1 = BarcodeTree(node.barcode, birth_lambda=self.birth_lambda)
            daughter1.tree.barcode.needs_repair = set()
            # oooh, recursion
            daughter1.simulate(simulation_time - t, root=False)
            node.add_child(daughter1.tree)
            daughter2 = BarcodeTree(node.barcode, birth_lambda=self.birth_lambda)
            daughter2.tree.barcode.needs_repair = set()
            daughter2.simulate(simulation_time - t, root=False)
            node.add_child(daughter2.tree)

        if root:
            # sequence alignment for leaf barcodes
            self.aln = MultipleSeqAlignment([])
            for i, leaf in enumerate(self.tree, 1):
                name = 'barcode{}'.format(i)
                leaf.name = name
                self.aln.append(SeqRecord(Seq(str(leaf.barcode).upper(), generic_dna), id=name, description=''))

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

    def n_leaves(self):
        return len(self.tree)





class BarcodeForest():
    '''
    simulate forest of BarcodeTree, all same parameters
    '''
    def __init__(self, barcode, birth_lambda, simulation_time=None, n=10, min_leaves=None):
        self.trees = []
        ct = 0
        while len(self.trees) < n:
            tree = BarcodeTree(barcode, birth_lambda, simulation_time=simulation_time)
            if min_leaves is None or tree.n_leaves() >= min_leaves:
                self.trees.append(tree)
                ct += 1
                print('trees simulated: {} of {}  \r'.format(ct, n), end='', flush=True)
        print()

    def editing_profile(self, file):
        '''plot profile of deletion frequency at each position over leaves'''
        plt.figure(figsize=(5,1))
        for i, tree in enumerate(self.trees):
            dat = []
            n_leaves = tree.n_leaves()
            for position, letter in enumerate(str(''.join(Barcode.v7))):
                if i == 0 and letter.islower():
                    plt.bar(position, 100, 1, facecolor='black', alpha=.2)
                dat.append(100*sum(str(leaf.barcode)[position] == '-' for leaf in tree.tree)/n_leaves)
            plt.plot(dat, alpha=.5, lw=2, clip_on=False)
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

    def summary_stats(self):
        # counter for the unique leaf genotypes
        genotypes = Counter([''.join(leaf.barcode.barcode) for leaf in self.tree])
        n_seqs = self.n_leaves()
        # weighted average deletion length
        mean_deletion_length = sum(len(run.group(0))*genotypes[genotype] for genotype in genotypes for run in re.finditer('-+', genotype))/sum(genotypes.values())
        return {'genotypes':len(genotypes), 'cells':n_seqs, 'mean deletion length':mean_deletion_length}

    def write_alignments(self, outbase):
        for i, tree in enumerate(self.trees, 1):
            tree.write_alignment('{}.{}.fasta'.format(outbase, i))
    def render(self, outbase):
        for i, tree in enumerate(self.trees, 1):
            tree.render('{}.{}.pdf'.format(outbase, i))

def main():
    '''do things, the main things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument('outbase', type=str, help='base name for plot and fasta output')
    parser.add_argument('--target_lambdas', type=float, nargs='+', default=[2**-n for n in range(10)], help='target cut poisson rates')
    parser.add_argument('--repair_lambda', type=float, default=10, help='repair poisson rate')
    parser.add_argument('--repair_deletion_probability', type=float, default=.1, help='probability of deletion during repair')
    parser.add_argument('--repair_deletion_lambda', type=float, default=1, help='poisson parameter for distribution of symmetric deltion about cut site(s) if deletion happens during repair')
    parser.add_argument('--birth_lambda', type=float, default=1, help='birth rate')
    parser.add_argument('--time', type=float, default=5, help='how much time to simulate')
    parser.add_argument('--min_leaves', type=int, default=200, help='condition on at least this many leaves')
    args = parser.parse_args()

    forest = BarcodeForest(Barcode(target_lambdas=args.target_lambdas,
                                   repair_lambda=args.repair_lambda,
                                   repair_deletion_probability=args.repair_deletion_probability,
                                   repair_deletion_lambda=args.repair_deletion_lambda),
                           birth_lambda=args.birth_lambda,
                           simulation_time=args.time,
                           min_leaves=args.min_leaves)

    forest.editing_profile(args.outbase + '.editing_profile.pdf')
    forest.write_alignments(args.outbase)
    forest.render(args.outbase)

    # print('summary statistic\tvalue')
    # for key, value in tree.summary_stats().items():
    #     print('{}\t{}'.format(key, value))

if __name__ == "__main__":
    main()
