"""
Read data from Stephen Quake's lab and convert to pickle file with List[ObservedAlignedSeq]
"""
from typing import List
import os
import sys
import csv
import argparse
import time
import logging
import pickle
import six

from clt_observer import ObservedAlignedSeq
from cell_state import CellState, CellTypeTree
from barcode_metadata import BarcodeMetadata
from allele_events import Event, AlleleEvents
from constants import BARCODE_QUAKE

def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--reads-file',
        type=str,
        default="/fh/fast/matsen_e/gestalt/quake/Split2_Annotated_Barcodes_edited.csv",
        help='Reads file from quake')
    parser.add_argument(
        '--out-obs-data',
        type=str,
        default="_output/worm2.pkl",
        help='file to store seq data')
    return parser.parse_args()

def _process_observed_indel_tract(
        primary_target: str,
        secondary_target: str,
        length: str,
        left_coord: str,
        right_coord: str,
        insertion: str,
        bcode_meta: BarcodeMetadata,
        threshold_left_coord: int = 5):
    primary_target = int(primary_target)
    secondary_target = int(secondary_target) if secondary_target != "" else int(primary_target)
    length = int(length)
    left_coord = int(left_coord)
    right_coord = int(right_coord)

    min_target = min(primary_target, secondary_target)
    max_target = max(primary_target, secondary_target)

    primary_target_cut_site = bcode_meta.abs_cut_sites[primary_target]
    secondary_target_cut_site = bcode_meta.abs_cut_sites[secondary_target]

    del_len = -length if length < 0 else 0

    if left_coord <= 0:
        start_pos = primary_target_cut_site + left_coord
    elif left_coord < threshold_left_coord:
        print("WARNING: weird left coord (it's positive?!)", left_coord)
        del_len += left_coord
        insertion += "A" * left_coord
        left_coord = 0
        start_pos = primary_target_cut_site
    else:
        # Maybe the cut site is actually for target `primary_target + 1`?
        raise ValueError("is the primary target correct??")

    for contained_target in range(min_target, max_target - 1):
        contained_cut_site = bcode_meta.abs_cut_sites[contained_target]
        if contained_cut_site > start_pos + del_len:
            print("secondary not covered...", contained_cut_site, start_pos + del_len)
            if contained_target == max_target - 2:
                omitted_insert_length = contained_cut_site - (start_pos + del_len)
                del_len += omitted_insert_length
                insertion += "A" * omitted_insert_length

    return Event(
        start_pos = start_pos,
        del_len = del_len,
        min_target = min_target,
        max_target = max_target,
        insert_str = insertion)

def parse_reads_file(file_name: str, bcode_meta: BarcodeMetadata):
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        primary_sites = next(reader)
        secondary_sites = next(reader)
        lengths = next(reader)
        left_coords = next(reader)
        right_coords = next(reader)
        insertions = next(reader)
        obs_allele_evts = list(zip(*[
            primary_sites,
            secondary_sites,
            lengths,
            left_coords,
            right_coords,
            insertions]))
        obs_allele_evts = obs_allele_evts[1:]
        allele_evts = [
                _process_observed_indel_tract(*allele_evt, bcode_meta=bcode_meta)
                for allele_evt in obs_allele_evts]

        num_organs = 0
        cell_states_dict = dict()
        obs_leaves = []
        observed_alleles = dict()
        for row in reader:
            organ_str = row[0][0]
            if organ_str not in cell_states_dict:
                cell_state = CellState(categorical=CellTypeTree(num_organs, rate=None))
                cell_states_dict[organ_str] = cell_state
                num_organs += 1
            cell_state = cell_states_dict[organ_str]

            allele_evts_for_seq = []
            for i, allele_evt_status in enumerate(row[1:]):
                if allele_evt_status == "1":
                    allele_evts_for_seq.append(allele_evts[i])
            obs_aligned_seq = ObservedAlignedSeq(
                allele_list = None,
                allele_events_list=[AlleleEvents(allele_evts_for_seq, bcode_meta.n_targets)],
                cell_state=cell_state,
                abundance=1)
            if str(obs_aligned_seq) not in observed_alleles:
                obs_leaves.append(obs_aligned_seq)
                observed_alleles[str(obs_aligned_seq)] = obs_aligned_seq
            else:
                print("abundance higher than one")
                observed_alleles[str(obs_aligned_seq)].abundance += 1

    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return obs_leaves, organ_dict

def main():
    args = parse_args()
    print(args)

    bcode_meta = BarcodeMetadata(
            unedited_barcode = BARCODE_QUAKE,
            num_barcodes = 1,
            # TODO: is this correct?
            cut_site = 6,
            # TODO: is this correct?
            crucial_pos_len=[6,6])

    obs_leaves, organ_dict = parse_reads_file(
            args.reads_file,
            bcode_meta)
    print("Number of leaves", len(obs_leaves))

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
