"""
Read data and create perfect phylogeny if possible
"""
from typing import List
import os
import sys
import csv
import numpy as np
import argparse
import time
import logging
import pickle
from pathlib import Path
import six

from barcode_metadata import BarcodeMetadata
from clt_observer import ObservedAlignedSeq
from cell_state import CellState, CellTypeTree
from allele_events import AlleleEvents
import data_binarizer

def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--obs-data',
        type=str,
        default="_output/fishies.pkl",
        help='pickled observed seq data')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.obs_data, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    obs_leaves = obs_data_dict["obs_leaves"]
    bcode_meta = obs_data_dict["bcode_meta"]

    processed_seqs, event_dicts, event_list = data_binarizer.binarize_observations(
            bcode_meta,
            obs_leaves)

    assert bcode_meta.num_barcodes == 1

    count_events = {k: 0 for k in event_dicts[0].keys()}
    character_sets = {k: [] for k in event_dicts[0].keys()}
    for seq_key, seq_dat in processed_seqs.items():
        evts_list = seq_dat[1]
        for evt in evts_list[0]:
            count_events[evt] += 1
            character_sets[evt].append(seq_key)
    event_count_tuples = [(evt, count) for evt, count in count_events.items()]
    sorted_event_count_tuples = [(None, -1)] + list(sorted(event_count_tuples, key = lambda x: x[1], reverse=True))

    char_tree_node_dict = {seq: 0 for seq in processed_seqs.keys()}
    char_node_links = {}
    for char_set_idx, (event, count) in enumerate(sorted_event_count_tuples):
        if char_set_idx == 0:
            continue
        char_set = character_sets[event]
        example_taxa = char_set[0]
        parent_node = char_tree_node_dict[example_taxa]
        for seq in char_set:
            if char_tree_node_dict[seq] != parent_node:
                print(sorted_event_count_tuples[parent_node])
                print("event", event)
                print("this seq", processed_seqs[seq])
                print("vs orig seq", processed_seqs[example_taxa])
                raise ValueError('No perfect phylogeny could be found %s' % seq)
            else:
                char_tree_node_dict[seq] = char_set_idx
        char_node_links[parent_node] = char_set_idx

    print("Found a perfect phylogeny!")

if __name__ == "__main__":
    main()
