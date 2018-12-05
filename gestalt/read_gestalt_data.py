"""
Read data from GESTALT and convert to pickle file with List[ObservedAlignedSeq]
"""
from typing import List
import random
import csv
import numpy as np
import argparse
import time
import logging
import six

from barcode_metadata import BarcodeMetadata
from clt_observer import ObservedAlignedSeq
from read_seq_data import process_event_format7B
from cell_state import CellState, CellTypeTree
from allele_events import AlleleEvents, Event
from anc_state import AncState
from constants import CONTROL_ORGANS
from constants import NO_EVENT_STRS
from constants import BARCODE_V6, NUM_BARCODE_V6_TARGETS

def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--reads-file',
        type=str,
        #default="/fh/fast/matsen_e/gestalt/fish_7B_UMI_collapsed_reads/fish_7B_UMI_collapsed_reads.txt",
        default="/fh/fast/matsen_e/gestalt/GSE81713_embryos_1_7/GSE81713_embryos_1_7.annotations.txt",
        help='Collapsed reads file: format 7B')
    parser.add_argument(
        '--reads-format',
        type=int,
        default=1,
        help='Format for reading this file')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/parse_gestalt.txt")
    parser.add_argument(
        '--out-obs-data',
        type=str,
        default="_output/fishies.pkl",
        help='file to store seq data')
    parser.add_argument(
        '--abundance-thres',
        type=int,
        default=0,
        help='Only include the alleles that have appeared at least this number of times')
    parser.add_argument(
        '--time',
        type=float,
        default=1,
        help='how much time we will say has elapsed')
    parser.add_argument(
        '--bcode-pad-length',
        type=int,
        default=20,
        help='barcode pad length')
    parser.add_argument(
        '--bcode-min-pos',
        type=int,
        default=122,
        help='barcode index offset')
    parser.add_argument(
        '--merge-thres',
        type=int,
        default=8,
        help='if events are within this many basepairs of each other, then merge as single event')
    return parser.parse_args()

def process_observed_seq_format7B(
        target_str_list: List[str],
        cell_state: CellState,
        bcode_meta: BarcodeMetadata,
        min_pos: int,
        abundance: int = 1,
        merge_thres: int = 1):
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

    # Assign the target to the one matching the cut site
    # Pad bad alignments with insertions and deletions
    cleaned_events = sorted(events, key=lambda ev: ev.start_pos)
    for i, evt in enumerate(cleaned_events):
        new_start_targ = evt.min_target
        new_end_targ = evt.max_target
        if evt.start_pos > bcode_meta.abs_cut_sites[evt.min_target]:
            # Start position is after the min target. Proposal is to shift the min cut target
            new_start_targ = evt.min_target + 1
        if evt.del_end < bcode_meta.abs_cut_sites[evt.max_target]:
            # End position is before the max target. Proposal is to shift the max cut target
            new_end_targ = evt.max_target - 1

        if new_start_targ > new_end_targ:
            # If we shifted such that the targets don't make sense, we need to start over.
            # This is probably a focal deletion that is not aligned with a cut site.
            # Determine the new target
            if new_start_targ > bcode_meta.n_targets - 1:
                new_target = new_end_targ
            elif new_end_targ < 0:
                new_target = 0
            else:
                cut_site1 = bcode_meta.abs_cut_sites[new_start_targ]
                cut_site2 = bcode_meta.abs_cut_sites[new_end_targ]
                min_dist1 = min(np.abs(evt.start_pos - cut_site1), np.abs(evt.del_end - cut_site1))
                min_dist2 = min(np.abs(evt.start_pos - cut_site2), np.abs(evt.del_end - cut_site2))
                new_target = new_start_targ if min_dist1 < min_dist2 else new_end_targ
            new_start_targ = new_target
            new_end_targ = new_target
            new_start_pos = min(evt.start_pos, bcode_meta.abs_cut_sites[new_start_targ])
            new_del_len = max(evt.del_end - new_start_pos, bcode_meta.abs_cut_sites[new_start_targ] - new_start_pos)
            # TODO: putting in a dummy insert sequence for now
            new_insert_str = evt.insert_str + "a" * (new_del_len - evt.del_len)
            event = Event(
                new_start_pos,
                new_del_len,
                new_start_targ,
                new_end_targ,
                new_insert_str)
        else:
            event = Event(
                evt.start_pos,
                evt.del_len,
                new_start_targ,
                new_end_targ,
                evt.insert_str)
        assert event.max_target < bcode_meta.n_targets
        cleaned_events[i] = event

    # Deal with multiple/compound events affecting the same targets.
    # We will merge these clashing events into a single event.
    non_clashing_events = cleaned_events[:1]
    for evt in cleaned_events[1:]:
        prev_evt = non_clashing_events[-1]
        _, max_deact_targ = prev_evt.get_min_max_deact_targets(bcode_meta)
        min_deact_targ, _ = evt.get_min_max_deact_targets(bcode_meta)
        do_merge_cause_close = max_deact_targ == min_deact_targ and ((evt.start_pos - prev_evt.del_end) <= merge_thres)
        #if max_deact_targ == min_deact_targ:
        #    print(evt.start_pos - prev_evt.del_end, merge_thres, do_merge_cause_close)
        do_merge_cause_impossible = prev_evt.max_target >= evt.min_target
        if do_merge_cause_close or do_merge_cause_impossible:
            new_omit_str_len = max(0, evt.start_pos - prev_evt.start_pos >= prev_evt.del_len)
            new_event = Event(
                    prev_evt.start_pos,
                    evt.start_pos - prev_evt.start_pos + evt.del_len,
                    prev_evt.min_target,
                    max(prev_evt.max_target, evt.max_target),
                    # TODO: we're putting a dummy insertion string when merging events
                    prev_evt.insert_str + evt.insert_str + "a" * new_omit_str_len)
            non_clashing_events[-1] = new_event
        else:
            non_clashing_events.append(evt)

    # Make sure the right trim length for the right-most target is not too long
    if len(non_clashing_events):
        last_evt = non_clashing_events[-1]
        if last_evt.max_target == bcode_meta.n_targets - 1:
            if last_evt.del_end > bcode_meta.orig_length:
                logging.info("last event overflow! shorten trim length!")
                print("last event overflow! shorten trim length!")
                non_clashing_events[-1] = Event(
                    last_evt.start_pos,
                    bcode_meta.orig_length - last_evt.start_pos,
                    last_evt.min_target,
                    last_evt.max_target,
                    last_evt.insert_str)

    #print("allelle", non_clashing_events)
    obs = ObservedAlignedSeq(
            None,
            [AlleleEvents(non_clashing_events)],
            cell_state,
            abundance=abundance)
    return obs

def parse_reads_file_format_GSE17(file_name,
                              bcode_meta: BarcodeMetadata,
                              bcode_min_pos: int,
                              max_read: int = None,
                              merge_thres: int = 1):
    """
    @param max_read: maximum number of alleles to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    num_organs = 0
    cell_states_dict = dict()
    all_alleles = []
    observed_alleles = dict()
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for i, row in enumerate(reader):
            organ_str = row[1].replace("17_", "")
            if organ_str not in CONTROL_ORGANS:
                # First process the organ
                if organ_str not in cell_states_dict:
                    cell_state = CellState(categorical=CellTypeTree(num_organs, rate=None))
                    cell_states_dict[organ_str] = cell_state
                    num_organs += 1
                cell_state = cell_states_dict[organ_str]
                abundance = int(row[2])
                obs_aligned_seq = process_observed_seq_format7B(
                    row[-1].split("_"),
                    cell_state,
                    bcode_meta,
                    bcode_min_pos,
                    abundance=abundance,
                    merge_thres=merge_thres)
                obs_key = str(obs_aligned_seq)
                if obs_key not in observed_alleles:
                    observed_alleles[obs_key] = obs_aligned_seq
                else:
                    obs = observed_alleles[obs_key]
                    obs.abundance += abundance
                if max_read is not None and len(all_alleles) == max_read:
                    break

    obs_alleles_list = list(observed_alleles.values())
    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return obs_alleles_list, organ_dict

def parse_reads_file_format_GSM(file_name,
                              bcode_meta: BarcodeMetadata,
                              bcode_min_pos: int,
                              target_hdr_fmt: str="target%d",
                              max_read: int = None,
                              merge_thres: int = 1):
    """
    @param max_read: maximum number of alleles to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    cell_state = CellState(categorical=CellTypeTree(0, rate=None))
    cell_states_dict = {"single_state": cell_state}
    all_alleles = []
    observed_alleles = dict()
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        target_start_idx = header.index(target_hdr_fmt % 1)
        for i, row in enumerate(reader):
            if row[1] != "PASS":
                continue

            # Now create allele representation
            obs_aligned_seq = process_observed_seq_format7B(
                [
                    row[target_start_idx + i]
                    for i in range(NUM_BARCODE_V6_TARGETS)
                ],
                cell_state,
                bcode_meta,
                bcode_min_pos,
                merge_thres=merge_thres)
            obs_key = str(obs_aligned_seq)
            if obs_key not in observed_alleles:
                observed_alleles[obs_key] = obs_aligned_seq
            else:
                obs = observed_alleles[obs_key]
                obs.abundance += 1
            if max_read is not None and len(all_alleles) == max_read:
                break

    obs_alleles_list = list(observed_alleles.values())
    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return obs_alleles_list, organ_dict

def parse_reads_file_format7B(file_name,
                              bcode_meta: BarcodeMetadata,
                              bcode_min_pos: int,
                              target_hdr_fmt: str="target%d",
                              max_read: int = None,
                              merge_thres: int = 1):
    """
    @param max_read: maximum number of alleles to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    num_organs = 0
    cell_states_dict = dict()
    all_alleles = []
    observed_alleles = dict()
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
                obs_aligned_seq = process_observed_seq_format7B(
                    [
                        row[target_start_idx + i]
                        for i in range(NUM_BARCODE_V6_TARGETS)
                    ],
                    cell_state,
                    bcode_meta,
                    bcode_min_pos,
                    merge_thres=merge_thres)
                obs_key = str(obs_aligned_seq)
                if obs_key not in observed_alleles:
                    observed_alleles[obs_key] = obs_aligned_seq
                else:
                    obs = observed_alleles[obs_key]
                    obs.abundance += 1
                if max_read is not None and len(all_alleles) == max_read:
                    break

    obs_alleles_list = list(observed_alleles.values())

    organ_dict = {}
    for organ_str, cell_type in cell_states_dict.items():
        organ_dict[str(cell_type)] = organ_str
    return obs_alleles_list, organ_dict

def merge_by_allele(obs_leaves: List[ObservedAlignedSeq]):
    """
    Merge observed sequences if they share the same allele.
    Ignore cell state
    """
    observed_alleles = dict()
    for obs in obs_leaves:
        obs_allele_key = obs.get_allele_str()
        if obs_allele_key not in observed_alleles:
            observed_alleles[obs_allele_key] = ObservedAlignedSeq(
                    obs.allele_list,
                    obs.allele_events_list,
                    None,
                    abundance = obs.abundance)
        else:
            matching_obs = observed_alleles[obs_allele_key]
            matching_obs.abundance += obs.abundance

    merged_obs_list = list(observed_alleles.values())
    return merged_obs_list

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Pad the barcode on the left and right in case there are long left/right deletions
    # at the ends of the barcode
    barcode_padded = (
            ('A' * args.bcode_pad_length + BARCODE_V6[0],)
            + BARCODE_V6[1:-1]
            + (BARCODE_V6[-1] + 'A' * args.bcode_pad_length,))

    bcode_meta = BarcodeMetadata(
            unedited_barcode = barcode_padded,
            num_barcodes = 1,
            # TODO: is this correct?
            cut_site = 6,
            # TODO: is this correct?
            crucial_pos_len=[6,6])

    if args.reads_format == 0:
        obs_leaves_cell_state, organ_dict = parse_reads_file_format7B(
            args.reads_file,
            bcode_meta,
            args.bcode_min_pos - args.bcode_pad_length,
            merge_thres=args.merge_thres)
    elif args.reads_format == 1:
        obs_leaves_cell_state, organ_dict = parse_reads_file_format_GSE17(
            args.reads_file,
            bcode_meta,
            args.bcode_min_pos - args.bcode_pad_length,
            merge_thres=args.merge_thres)
    elif args.reads_format == 2:
        obs_leaves_cell_state, organ_dict = parse_reads_file_format_GSM(
            args.reads_file,
            bcode_meta,
            args.bcode_min_pos - args.bcode_pad_length,
            merge_thres=args.merge_thres)
    else:
        raise ValueError("huh")
    obs_leaves_cell_state = [obs for obs in obs_leaves_cell_state if obs.abundance >= args.abundance_thres]
    logging.info("Number of uniq allele cell state pairs %d", len(obs_leaves_cell_state))
    print("Number of uniq allele cell state pairs", len(obs_leaves_cell_state))

    obs_leaves = merge_by_allele(obs_leaves_cell_state)

    # Check trim length assignments
    # Check all indels are disjoin in terms of what targets they deactivate
    evt_to_obs = {}
    for obs in obs_leaves:
        anc_state = AncState.create_for_observed_allele(obs.allele_events_list[0], bcode_meta)
        for evt in anc_state.indel_set_list:
            if evt not in evt_to_obs:
                evt_to_obs[evt] = [anc_state]
            else:
                evt_to_obs[evt].append(anc_state)
    for obs in obs_leaves:
        try:
            for evt in obs.allele_events_list[0].events:
                evt.get_trim_lens(bcode_meta)
        except AssertionError as e:
            print(e)
            print(obs.allele_events_list[0])
            print(bcode_meta.right_max_trim)
            for evt in obs.allele_events_list[0].events:
                print(evt.get_trim_lens(bcode_meta))
        anc_state = AncState.create_for_observed_allele(obs.allele_events_list[0], bcode_meta)
        for i, evt in enumerate(anc_state.indel_set_list):
            if i == 0:
                continue
            prev_evt = anc_state.indel_set_list[i - 1]
            if prev_evt.max_deact_target == evt.min_deact_target and (evt.min_deact_target == evt.min_target or prev_evt.max_deact_target == prev_evt.max_target):
                print(
                    "There are clashing events in the allele with ancstate %s. Distance between clashing events: %d."
                    % (anc_state, evt.start_pos - prev_evt.del_end))
                if evt.min_deact_target != evt.min_target:
                    irreversibility_checks = [prev_evt in a.indel_set_list for a in evt_to_obs[evt]]
                else:
                    irreversibility_checks = [evt in a.indel_set_list for a in evt_to_obs[prev_evt]]
                if len(irreversibility_checks) > 2 and not all(irreversibility_checks):
                    raise ValueError("nope. clashing events not preserving irreversibility (though crazy homoplasy could also be occurring), %s" % irreversibility_checks)

    # Save the observed data
    with open(args.out_obs_data, "wb") as f:
        out_dict = {
            "bcode_meta": bcode_meta,
            "obs_leaves": obs_leaves,
            "obs_leaves_by_allele_cell_state": obs_leaves_cell_state,
            "organ_dict": organ_dict,
            "time": args.time,
        }
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
