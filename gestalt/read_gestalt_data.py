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
import logging
import pickle
from pathlib import Path
import six

from barcode_metadata import BarcodeMetadata
from clt_observer import ObservedAlignedSeq
from read_seq_data import process_event_format7B
from cell_state import CellState, CellTypeTree
from allele_events import AlleleEvents, Event
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
        '--log-file',
        type=str,
        default="_output/parse_gestalt.txt")
    parser.add_argument(
        '--out-obs-data',
        type=str,
        default="_output/fishies.pkl",
        help='file to store seq data')
    return parser.parse_args()

def process_observed_seq_format7B(
        target_str_list: List[str],
        cell_state: CellState,
        bcode_meta: BarcodeMetadata,
        min_pos: int = 122):
    """
    Converts new format allele to python repr allele

    @param target_str_list: targets with events encoded in 7B format
    @param cell_state: the cell type associated with this allele
    @return ObservedAlignedSeq
    """
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
        process_event_format7B(event_str, min_targ, max_targ, min_pos=min_pos)
        for event_str, (min_targ, max_targ) in evt_target_dict.items() if event_str not in NO_EVENT_STRS
    ]
    events = sorted(events, key=lambda ev: ev.start_pos)

    print(bcode_meta.abs_cut_sites)
    # Deal with multiple/compound events affecting same min target
    cleaned_events = events[:1]
    for evt in events[1:]:
        prev_evt = cleaned_events[-1]
        if prev_evt.min_target == evt.min_target:
            new_event = Event(
                    prev_evt.start_pos,
                    # TODO: this is hack. not correct right now
                    evt.del_len + prev_evt.del_len,
                    prev_evt.min_target,
                    evt.max_target,
                    # TODO: this is hack. not correct right now
                    prev_evt.insert_str + evt.insert_str)
            assert prev_evt.max_target <= evt.max_target
            cleaned_events[-1] = new_event
        else:
            cleaned_events.append(evt)

    # Deal with weird alignment of indel events vs barcode cut sites
    # Pad bad alignments with insertions and deletions
    print("orig", events)
    print("semi cl", cleaned_events)
    for i, evt in enumerate(cleaned_events):
        new_start_targ = evt.min_target
        new_end_targ = evt.max_target
        if evt.start_pos > bcode_meta.abs_cut_sites[evt.min_target]:
            new_start_targ = evt.min_target + 1
        if evt.start_pos + evt.del_len < bcode_meta.abs_cut_sites[evt.max_target]:
            new_end_targ = evt.max_target - 1

        if new_start_targ > new_end_targ:
            new_start_targ = evt.min_target
            new_end_targ = min(evt.min_target + 1, bcode_meta.n_targets - 1)
            dist_to_start = np.abs(bcode_meta.abs_cut_sites[new_start_targ] - evt.start_pos)
            dist_to_end = np.abs(bcode_meta.abs_cut_sites[new_end_targ] - evt.start_pos)
            if dist_to_start > 2 * dist_to_end:
                new_start_targ = new_end_targ

            new_cut_site = bcode_meta.abs_cut_sites[new_start_targ]
            new_start_pos = min(evt.start_pos, new_cut_site)
            cleaned_events[i] = Event(
                    new_start_pos,
                    new_cut_site - new_start_pos,
                    new_start_targ,
                    new_start_targ,
                    evt.insert_str + "A" * (new_cut_site - new_start_pos))
        else:
            cleaned_events[i] = Event(
                    evt.start_pos,
                    evt.del_len,
                    new_start_targ,
                    new_end_targ,
                    evt.insert_str)

    print("semi-semi cl", cleaned_events)
    i = 0
    while i < len(cleaned_events) - 1:
        min_deact_target, max_deact_target = bcode_meta.get_min_max_deact_targets(cleaned_events[i])
        #min_deact_target_nxt, max_deact_target_nxt = bcode_meta.get_min_max_deact_targets(cleaned_events[i + 1])
        if max_deact_target > cleaned_events[i + 1].min_target:
            print("clash...", cleaned_events[i], cleaned_events[i + 1])
            print(min_deact_target, max_deact_target)
            raise ValueError("max deactivated target is larger than the min target for the next event")
            #evt = cleaned_events[i]
            #if evt.del_len:
            #    cleaned_events[i] = Event(
            #        evt.start_pos,
            #        evt.del_len - 1,
            #        evt.min_target,
            #        evt.max_target,
            #        evt.insert_str)
        else:
            i = i + 1

    target_tracts = []
    for evt in cleaned_events:
        min_deact_target, max_deact_target = bcode_meta.get_min_max_deact_targets(evt)
        target_tracts.append((min_deact_target, evt.min_target, evt.max_target, max_deact_target))
    print("target tract", target_tracts)

    print("clean", cleaned_events)
    print('----------------------------------------')
    return ObservedAlignedSeq(None, [AlleleEvents(cleaned_events)], cell_state, abundance=1)

def parse_reads_file_format7B(file_name,
                              bcode_meta: BarcodeMetadata,
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
        for i, row in enumerate(reader):
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
                    cell_state,
                    bcode_meta)
                all_alleles.append(allele_events)
                if max_read is not None and len(all_alleles) == max_read:
                    break

    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return all_alleles, organ_dict

def filter_for_common_alleles(all_alleles: List[ObservedAlignedSeq], threshold = 5):
    count_dict = {}
    for allele in all_alleles:
        key = (tuple(allele.allele_events_list), allele.cell_state)
        if key in count_dict:
            count_dict[key][0] += 1
        else:
            count_dict[key] = [1, allele]

    common_alleles = []
    for key, (count, allele) in count_dict.items():
        if count > threshold:
            common_alleles.append(allele)
            num_events = len(allele.allele_events_list[0].events)
            num_min_targets = np.unique([e.min_target for e in allele.allele_events_list[0].events]).size
            assert num_events == num_min_targets

    return common_alleles

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    bcode_meta = BarcodeMetadata(
            unedited_barcode = BARCODE_V7,
            num_barcodes = 1,
            # TODO: is this correct?
            cut_site = 6,
            # TODO: is this correct?
            crucial_pos_len=[6,6])

    obs_leaves, organ_dict = parse_reads_file_format7B(args.reads_file, bcode_meta)
    obs_leaves = filter_for_common_alleles(obs_leaves)
    logging.info("Number of leaves %d", len(obs_leaves))

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
