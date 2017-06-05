#! /usr/bin/env python
# -*- coding: utf-8 -*-

import scipy, argparse
from scipy.stats import expon
from numpy.random import choice, randint
import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde

class Barcode():
    def __init__(self, lambdas, repair_time=1., target_length=100, repair_deletion=1):
        self.lambdas = scipy.array(lambdas)
        self.n_targets = len(lambdas)
        self.repair_time = repair_time
        self.target_length = target_length
        self.target_lengths = [target_length] * self.n_targets
        self.repair_deletion = repair_deletion
        if repair_deletion != 1:
            raise NotImplementedError('must use repair_deletion=1')
        self.barcode = [list(range(target_length)) for _ in range(self.n_targets)]

    def simulate(self, time):

        lambda_total = self.lambdas.sum()
        new_lambdas = self.lambdas
        relative_rates = scipy.array([lambda_/lambda_total for lambda_ in self.lambdas])

        t = 0
        last_edit = None
        while True:
            t += expon.rvs(scale=1/lambda_total)
            if t > time: break
            target_index = choice(self.n_targets, p=relative_rates)
            # preference to edit ~ 75% position
            profile = scipy.array([1] * self.target_lengths[target_index])
            if profile.shape[0] > 1:
                profile[round(.75*profile.shape[0])-1] = 50
                # profile = gaussian_kde([x for x in range(len(profile)) for _ in range(profile[x])], .1).pdf(list(range(profile.shape[0])))
            profile = profile/profile.sum()

            position_index = self.barcode[target_index].pop(choice(self.target_lengths[target_index], p=profile))
            self.target_lengths[target_index] -= 1
            if last_edit is not None and t - last_edit[0] < self.repair_time:
                left = min((last_edit[1:], (target_index, position_index)))
                right = max((last_edit[1:], (target_index, position_index)))
                if left[0] == right[0]:
                    self.barcode[left[0]] = [x for x in self.barcode[left[0]] if x < left[1] or x > right[1]]
                else:
                    self.barcode[left[0]] = [x for x in self.barcode[left[0]] if x < left[1]]
                    self.barcode[right[0]] = [x for x in self.barcode[right[0]] if x > right[1]]
                    for i in range(left[0] + 1, right[0]):
                        self.barcode[i] = []
            for i, target in enumerate(self.barcode):
                self.target_lengths[i] = len(target)
                new_lambdas[i] = self.lambdas[i]*(self.target_lengths[i]/self.target_length)
            lambda_total = new_lambdas.sum()
            if lambda_total <= 0:
                return
            relative_rates = scipy.array([lambda_/lambda_total for lambda_ in new_lambdas])
            last_edit = (t, target_index, position_index)

    def __repr__(self):
        return str(self.barcode)


# class Barcodes(Barcode):
#     '''container of Barcode'''
#     def __init__(self, lambdas, repair_time=1, target_length=100, repair_deletion=1, n_barcodes=100):
#         Barcode.__init__(self,
#                          lambdas,
#                          repair_time=repair_time,
#                          target_length=target_length,
#                          repair_deletion=repair_deletion)
#         self.n_barcodes = n_barcodes
#         self.target_length = target_length
#         self.barcode = [Barcode(lambdas,
#                                  repair_time=self.repair_time,
#                                  target_length=self.target_length,
#                                  repair_deletion=self.repair_deletion) for _ in range(self.n_barcodes)]
#     def simulate(self, time):
#         for barcode in self.barcode:
#             barcode.simulate(time)
#     def plot(self, plot_file):
#         dat = []
#         fig = plt.figure(figsize=(10,3))
#         for target in range(self.n_targets):
#             if target > 0:
#                 plt.axvline(x=target*self.target_length, color='black', alpha=.2, lw=3)
#             for position in range(self.target_length):
#                 dat.append(sum(position not in barcode.barcode[target] for barcode in self.barcode))
#         plt.plot(dat, color='red', lw=2, clip_on=False)
#         plt.xlim(0, self.n_targets*self.target_length)
#         plt.ylim(0, self.n_barcodes)
#         fig.set_tight_layout(True)
#         plt.savefig(plot_file)
#         print(dat)

def main():

    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument('outfile', type=str, help='plot file name')
    parser.add_argument('--repair_time', type=float, default=0, help='time for NHEJR')
    args = parser.parse_args()

    target_length=23
    n_barcodes=50
    sim_time = 2
    lambdas = [2**-n for n in range(10)]
    n_targets = len(lambdas)

    barcodes = [Barcode(lambdas=lambdas,
                        repair_time=args.repair_time,
                        target_length=target_length) for _ in range(n_barcodes)]
    for barcode in barcodes:
        barcode.simulate(sim_time)

    dat = []
    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
    plt.subplot(gs[0])
    for target in range(n_targets):
        if target > 0:
            plt.axvline(x=target*target_length, color='black', alpha=.2, lw=3)
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
    ax.axis('off')
    plt.xlim(0, n_targets*target_length)
    fig.set_tight_layout(True)
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
