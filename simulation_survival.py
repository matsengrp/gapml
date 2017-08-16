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
    '''GESTALT target array with PAM sequences'''
    def __init__(self, lambdas, repair_lambda, repair_deletion_lambda=.1):
        # v7 barcode from GESTALT paper Table S4
        self.barcode = [  'cg', 'GATACGATACGCGCACGCTATGG',
                        'agtc', 'GACACGACTCGCGCATACGATGG',
                        'agtc', 'GATAGTATGCGTATACGCTATGG',
                        'agtc', 'GATATGCATAGCGCATGCTATGG',
                        'agtc', 'GAGTCGAGACGCTGACGATATGG',
                        'agtc', 'GCTACGATACACTCTGACTATGG',
                        'agtc', 'GCGACTGTACGCACACGCGATGG',
                        'agtc', 'GATACGTAGCACGCAGACTATGG',
                        'agtc', 'GACACAGTACTCTCACTCTATGG',
                        'agtc', 'GATATGAGACTCGCATGTGATGG',
                        'ga']
        self.n_targets = len(self.v7)/2 - 1
        if len(lambdas) != self.n_targets:
            raise ValueError('must give {} lambdas'.format(self.n_targets))
        self.lambdas = scipy.array(lambdas)
        self.repair_lambda = repair_lambda
        self.target_length = len(self.v7[1])
        self.target_lengths = [target_length] * self.n_targets
        self.repair_deletion_lambda = repair_deletion_lambda

    def simulate(self, max_cuts):
        needs_repair = False
        last_target = None
        last_deletion_length = None
        tot_cuts = 0
        # cut position
        position_index = self.target_length - 6
        while tot_cuts < max_cuts:
            if not needs_repair:
                target_index = choice(self.n_targets, p=self.lambdas/self.lambdas.sum())
            else:
                repair_and_target_lambdas = scipy.concatenate([self.lambdas, [self.repair_lambda]])
                target_index = choice(self.n_targets + 1, p=repair_and_target_lambdas/repair_and_target_lambdas.sum())
            if target_index < self.n_targets:
                tot_cuts += 1
                deletion_length = poisson.rvs(self.repair_deletion_lambda)
                ###
                self.barcode[target_index] = [x for x in range(self.target_lengths[target_index]) if x <= position_index - deletion_length or x > position_index + deletion_length]
                self.target_lengths[target_index] -= 2*deletion_length

                if needs_repair:
                    # we encountered a double cut
                    left = min(((last_target, position_index - last_deletion_length), (target_index, position_index - deletion_length)))
                    right = max(((last_target, position_index + last_deletion_length), (target_index, position_index + deletion_length)))
                    if left[0] == right[0]:
                        self.barcode[left[0]] = [x for x in self.barcode[left[0]] if x < left[1] or x > right[1]]
                    else:
                        self.barcode[left[0]] = [x for x in self.barcode[left[0]] if x < left[1]]
                        self.barcode[right[0]] = [x for x in self.barcode[right[0]] if x > right[1]]
                        for i in range(left[0] + 1, right[0]):
                            self.barcode[i] = []

            for i, target in enumerate(self.barcode):
                self.target_lengths[i] = len(target)
                # if shorter, send to zero
                if self.target_lengths[i] < self.target_length:
                    self.lambdas[i] = 0 #self.lambdas[i]*(self.target_lengths[i]//self.target_length)
            if self.lambdas.sum() == 0:
                return
            last_target = target_index
            last_deletion_length = deletion_length
            needs_repair = target_index < self.n_targets

    def __repr__(self):
        return str(self.barcode)

def main():

    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument('outfile', type=str, help='plot file name')
    args = parser.parse_args()


    n_barcodes=50
    lambdas = [2**-n for n in range(3)] + [2**-n for n in range(5,12)]
    n_targets = len(lambdas)

    barcodes = [
        Barcode(
            lambdas=lambdas,
            repair_lambda=.5,
            repair_deletion_lambda=1)
        for _ in range(n_barcodes)
    ]

    for barcode in barcodes:
        barcode.simulate(scipy.random.choice(2) + 1)

    dat = []
    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
    plt.subplot(gs[0])
    for target in range(n_targets):
        if target > 0:
            plt.axvline(x=target*barcode.target_length, color='black', alpha=.2, lw=3)
        for position in range(target_length):
            dat.append(sum(position not in barcode.barcode[target] for barcode in barcodes))
    dat = scipy.array(dat)
    plt.plot(100*dat/n_barcodes, color='red', lw=2, clip_on=False)
    plt.xlim(0, n_targets*target_length)
    plt.ylim(0, None)
    plt.ylabel('Editing (%)')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

    ax = plt.subplot(gs[1])
    for i, barcode in enumerate(barcodes):
        for j, target in enumerate(barcode.barcode):
            for k in range(target_length):
                if k not in target:
                    plt.barh(i, 1, left=j*target_length+k, color='red')
        plt.axhline(y=i-.5, color='black', lw=.1)
    plt.axhline(y=i+1-.5, color='black', lw=.1)
    ax.axis('off')
    plt.xlim(0, n_targets*target_length)
    fig.set_tight_layout(True)
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
