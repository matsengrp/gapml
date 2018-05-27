"""
Read data from GESTALT and convert to pickle file with List[ObservedAlignedSeq]
"""
from typing import List
import os
import sys
import csv
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import logging
import pickle
from pathlib import Path
import six

from barcode_metadata import BarcodeMetadata
from clt_observer import ObservedAlignedSeq
from read_seq_data import process_event_format7B
from cell_state import CellState, CellTypeTree
from allele_events import AlleleEvents
from constants import CONTROL_ORGANS
from constants import NO_EVENT_STRS
from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS

def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--reads-file',
        type=str,
        default="/fh/fast/matsen_e/gestalt/fish_7B_UMI_collapsed_reads/fish_7B_UMI_collapsed_reads.txt",
        help='Collapsed reads file: format 7B')
    parser.add_argument(
        '--out-obs-data',
        type=str,
        default="_output/fishies.pkl",
        help='file to store seq data')
    return parser.parse_args()

def process_observed_seq_format7B(target_str_list: List[str], cell_state: CellState):
    """
    Converts new format allele to python repr allele

    @param target_str_list: targets with events encoded in 7B format
    @param cell_state: the cell type associated with this allele
    @return ObservedAlignedSeq
    """
    #print(target_str_list)
    # Find list of targets for each event
    evt_target_dict = {}
    for targ_idx, targ_str in enumerate(target_str_list):
        target_evt_strs = targ_str.split("&")
        for evt_str in target_evt_strs:
            if evt_str not in evt_target_dict:
                evt_target_dict[evt_str] = (targ_idx, targ_idx)
            else:
                min_targ, max_targ = evt_target_dict[evt_str]
                evt_target_dict[evt_str] = (min(min_targ, targ_idx), max(max_targ, targ_idx))
    events = [
        process_event_format7B(event_str, min_targ, max_targ) for event_str, (min_targ, max_targ) in evt_target_dict.items() if event_str not in NO_EVENT_STRS
    ]
    events = sorted(events, key=lambda ev: ev.min_target)

    #####
    # Jean's experiments to see how clean this data is...
    # We need the same cut site across all barcodes and we need no compound events...
    #####
    #for i in range(10):
    #    has_zero_del = False
    #    for ev in events:
    #        if ev.min_target == i and ev.del_len == 0:
    #            has_zero_del = True
    #            break
    #    if has_zero_del:
    #        match_events = [ev for ev in events if ev.min_target == i]
    #        if len(match_events) == 1:
    #            print(match_events)

    #for i in range(len(events) - 1):
    #    print(events[i].max_target >= events[i+1].min_target)

    # TODO: This will have to do for now...
    cleaned_events = events

    return ObservedAlignedSeq(None, AlleleEvents(cleaned_events), cell_state, abundance=1)

def parse_reads_file_format7B(file_name,
                              target_hdr_fmt="target%d",
                              max_read=None):
    """
    @param max_read: maximum number of alleles to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    num_organs = 0
    cell_states_dict = dict()
    all_alleles = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        target_start_idx = header.index(target_hdr_fmt % 1)
        for row in reader:
            organ_str = row[0]
            if organ_str not in CONTROL_ORGANS:
                # First process the organ
                if organ_str not in cell_states_dict:
                    cell_state = CellState(categorical=CellTypeTree(num_organs, rate=None))
                    cell_states_dict[organ_str] = cell_state
                    num_organs += 1
                cell_state = cell_states_dict[organ_str]

                # Now create allele representation
                allele_events = process_observed_seq_format7B(
                    [
                        row[target_start_idx + i]
                        for i in range(NUM_BARCODE_V7_TARGETS)
                    ],
                    cell_state)
                all_alleles.append(allele_events)
                if max_read is not None and len(all_alleles) == max_read:
                    break

    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return all_alleles, organ_dict

def main():
    args = parse_args()

    bcode_meta = BarcodeMetadata(
            unedited_barcode = BARCODE_V7,
            num_barcodes = 1,
            # TODO: is this correct?
            cut_site = 6,
            # TODO: is this correct?
            crucial_pos_len=[6,6])

    obs_leaves, organ_dict = parse_reads_file_format7B(args.reads_file)

    # Save the observed data
    with open(args.out_obs_data, "wb") as f:
        out_dict = {
            "bcode_meta": bcode_meta,
            "obs_leaves": obs_leaves,
            "organ_dict": organ_dict,
        }
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
