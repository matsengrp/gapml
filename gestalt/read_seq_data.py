import csv
import argparse
from typing import List

from barcode_events import Event
from barcode_events import BarcodeEvents
from all_reads import CellReads, CellRead
from cell_state import CellTypeTree
from constants import BARCODE_V7
from constants import BARCODE_V7_LEN
from constants import CONTROL_ORGANS
from constants import NO_EVENT_STRS
from constants import NUM_BARCODE_V7_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--reads_file',
        type=str,
        default="/fh/fast/matsen_e/gestalt/fish_7B_UMI_collapsed_reads/fish_7B_UMI_collapsed_reads.txt",
        help='Collapsed reads file: format 7B')
    return parser.parse_args()

def process_event_format7B(event_str: str, min_target: int, max_target: int):
    """
    Takes a single event string and creates an event object
    Right now processes events in a super dumb way.
    There are no complex events grouped in a single event
    (e.g. there are no simultaneous deletions and insertions right now)
    TODO: make this better? make the sequence file better?
    """
    event_split = event_str.split("+")
    event_type_str = event_split[0][-1]
    event_pos = int(event_split[1])
    if event_type_str == "D":
        del_len = int(event_split[0][:-1])
        return Event(event_pos, del_len, min_target=min_target, max_target=max_target, insert_str="")
    elif event_type_str == "I":
        return Event(event_pos, del_len=0, min_target=min_target, max_target=max_target, insert_str=event_split[2])
    else:
        raise ValueError("Unrecognized event: %s" % event_str)

def process_barcode_format7B(target_str_list: List[str], cell_type_tree: CellTypeTree):
    """
    Converts new format barcode to python repr barcode

    @param target_str_list: targets with events encoded in 7B format
    @param cell_type_tree: the cell type associated with this barcode
    @return barcode in event-encoded format
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
        process_event_format7B(event_str, min_targ, max_targ) for event_str, (min_targ, max_targ) in evt_target_dict.items() if event_str not in NO_EVENT_STRS
    ]

    return CellRead(BarcodeEvents(events), cell_type_tree)

def parse_reads_file_format7B(file_name,
                              target_hdr_fmt="target%d",
                              max_read=None):
    """
    @param max_read: maximum number of barcodes to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    num_organs = 0
    organ_cell_types = dict()
    all_barcodes = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        target_start_idx = header.index(target_hdr_fmt % 1)
        for row in reader:
            organ_str = row[0]
            if organ_str not in CONTROL_ORGANS:
                # First process the organ
                if organ_str not in organ_cell_types:
                    cell_type_tree = CellTypeTree(num_organs, rate=None, probability=None)
                    organ_cell_types[organ_str] = cell_type_tree
                    num_organs += 1
                cell_type_tree = organ_cell_types[organ_str]

                # Now create barcode representation
                barcode_events = process_barcode_format7B(
                    [
                        row[target_start_idx + i]
                        for i in range(NUM_BARCODE_V7_TARGETS)
                    ],
                    cell_type_tree)
                all_barcodes.append(barcode_events)
                if max_read is not None and len(all_barcodes) == max_read:
                    break

    organ_dict = {}
    for organ_str, cell_type in organ_cell_types.items():
        organ_dict[cell_type.cell_type] = organ_str
    return CellReads(all_barcodes, organ_dict)

def parse_reads_file_newformat(file_name,
                            organ_data_idx=1,
                            target_data_idx=2,
                            max_read=None):
    """
    @param max_read: maximum number of barcodes to read (for debugging purposes)

    Right now, Aaron's file outputs all the events associated with a target.
    This means for inter-target events, it will appear multiple times on that row.
    e.g. target1 33D+234, target2 33E+234
    """
    num_organs = 0
    organ_cell_types = dict()
    all_barcodes = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            organ_str = row[organ_data_idx]
            if organ_str not in CONTROL_ORGANS:
                # First process the organ
                if organ_str not in organ_cell_types:
                    cell_type_tree = CellTypeTree(num_organs, rate=None, probability=None)
                    organ_cell_types[organ_str] = cell_type_tree
                    num_organs += 1
                cell_type_tree = organ_cell_types[organ_str]

                # Now create barcode representation
                barcode_data = row[target_data_idx].split("_")
                assert(len(barcode_data) == NUM_BARCODE_V7_TARGETS)
                barcode_events = process_barcode_format7B(
                    barcode_data,
                    cell_type_tree)
                all_barcodes.append(barcode_events)
                if max_read is not None and len(all_barcodes) == max_read:
                    break

    organ_dict = {}
    for organ_str, cell_type in organ_cell_types.items():
        organ_dict[cell_type.cell_type] = organ_str
    return CellReads(all_barcodes, organ_dict)

def main():
    args = parse_args()
    cell_reads = parse_reads_file_format7B(args.reads_file)


if __name__ == "__main__":
    main()
