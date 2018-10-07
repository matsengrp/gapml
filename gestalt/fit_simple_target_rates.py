"""
Fit only cut rate parameters via simple average
"""
import sys
import six
import os
import argparse
import logging
import numpy as np
import random
from typing import Dict

from barcode_metadata import BarcodeMetadata


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.set_defaults(tot_time_known=True)
    args = parser.parse_args(args)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
    bcode_meta = obs_data_dict["bcode_meta"]
    obs_data = obs_data_dict["obs_leaves"]
    assert bcode_meta.num_barcodes == 1

    all_events = [o.allele_events_list[0].events for o in obs_data]
    uniq_events = set([evt for evts in all_events for evt in evts])
    targ_used = np.zeros(bcode_meta.n_targets)
    for evt in uniq_events:
        if evt.min_target != evt.max_target:
            targ_used[evt.min_target] += 1
            targ_used[evt.max_target] += 1
        else:
            targ_used[evt.min_target] += 1
    targ_use_rate = targ_used/np.sum(targ_used)
    print([a for a in targ_use_rate])

if __name__ == "__main__":
    main()
