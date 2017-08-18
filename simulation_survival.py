#! /usr/bin/env python
# -*- coding: utf-8 -*-

import scipy, argparse
from scipy.stats import expon, poisson
from numpy.random import choice, randint
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde

class Barcode():
    '''GESTALT target array with spacer sequences'''
    def __init__(self, lambdas, repair_lambda, repair_deletion_lambda=.1):
        # v7 barcode from GESTALT paper Table S4
        self.barcode0 = (  'cg', 'GATACGATACGCGCACGCTATGG',
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
        # number of targets
        self.n_targets = (len(self.barcode0) - 1)//2
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(self.barcode0)
        if len(lambdas) != self.n_targets:
            raise ValueError('must give {} lambdas'.format(self.n_targets))
        # poisson rates for each target
        self.lambdas = scipy.array(lambdas)
        # poisson rate for repair process
        self.repair_lambda = repair_lambda
        # poisson parameter for how much (~symmetric) deletion about cut sites happens when repair happens
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
        assert len(''.join(self.barcode)) == len(''.join(self.barcode0))

    def repair(self):
        '''
        repair DSB cuts
        - if self.needs_repair is empty, do nothing
        - if self.needs_repair contains one element, create a repair scar in that target (index), with specified length in both directions
        - it self.needs_repair contains two or more elements, create an inter-target deletion, randomly choosing two indices (with replacement)
        - deletions can span target boundaries
        - any targets that are modified get their lambda sent to zero
        '''
        if len(self.needs_repair) == 0:
            return
        # a random draw for symmetric deletion about cut site(s)
        deletion_length = poisson.rvs(self.repair_deletion_lambda)
        # choose a random pair of targets among those that need repair (with replacement)
        target1, target2 = choice(list(self.needs_repair), 2)
        # create deletion
        self.delete(target1, target2, deletion_length)
        # update lambdas
        for target in range(self.n_targets):
            index = 1 + 2*target
            # NOTE: this is more robust than checking for gap characters, since we might later model insertions
            if self.barcode[index] != self.barcode0[index]:
                self.lambdas[target] = 0
        # update which targets still need repair
        self.needs_repair = {target for target in self.needs_repair if target < min(target1, target2) or target > max(target1, target2)}

    def simulate(self, n_repairs):
        '''
        simulate a specified number of repair events
        NOTE: possible that repair event leaves no deletion
        '''
        # counts how many repair events have occured
        repairs = 0
        while repairs < n_repairs:
            # line up the target cut rates and the repair process rate, so they can race!
            event_lambdas = scipy.concatenate([self.lambdas, [self.repair_lambda]])
            event = choice(self.n_targets + 1, p=event_lambdas/event_lambdas.sum())
            # the last event represents the repair process
            if event == self.n_targets:
                self.repair()
                repairs += 1 # NOTE: this gets incremented even if repair did nothing because of an empty self.needs_repair (to be consistent with a fixed time)
            else:
                self.needs_repair.add(event) # NOTE: if this target already needed repair, this doesn't change anything

    def __repr__(self):
        return str(''.join(self.barcode))

def main():
    '''do things'''
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument('outfile', type=str, help='plot file name')
    parser.add_argument('--n', type=int, default=100, help='how many barcodes to simulate')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[2**-n for n in range(10)], help='target cut poisson rates')
    parser.add_argument('--repair_lambda', type=float, default=1., help='repair poisson rate')
    parser.add_argument('--repair_deletion_lambda', type=float, default=1., help='poisson parameter for distribution of symmetric deltion about cut site(s) during repair')
    parser.add_argument('--n_repair', type=int, default=10, help='how many repair events to simulate')
    args = parser.parse_args()

    barcodes = [Barcode(lambdas=args.lambdas,
                        repair_lambda=args.repair_lambda,
                        repair_deletion_lambda=args.repair_deletion_lambda)
                for _ in range(args.n)]

    for barcode in barcodes:
        # NOTE: with multiple barcodes, is it weird to be fixing the number of events,
        #       rather than the time? We don't get any variation in the number of events this way
        barcode.simulate(args.n_repair)
        print(barcode)

    bc_unedited = str(''.join(barcodes[0].barcode0))

    dat = []
    fig = plt.figure(figsize=(4,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
    plt.subplot(gs[0])
    for position, letter in enumerate(bc_unedited):
        if letter.islower():
            plt.bar(position, 100, 1, facecolor='black', alpha=.2)
        dat.append(sum(str(barcode)[position] == '-' for barcode in barcodes))
    plt.plot(100*scipy.array(dat)/args.n, color='red', lw=2, clip_on=False)
    plt.xlim(0, len(bc_unedited))
    plt.ylim(0, 100)
    plt.ylabel('Editing (%)')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

    ax = plt.subplot(gs[1])
    for i, barcode in enumerate(barcodes):
        for j, letter in enumerate(str(barcode)):
            if letter == '-':
                plt.barh(i, 1, left=j, color='red')
        plt.axhline(y=i-.5, color='black', lw=.1)
    plt.axhline(y=i+1-.5, color='black', lw=.1)
    ax.axis('off')
    plt.xlim(0, len(bc_unedited))
    fig.set_tight_layout(True)
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
