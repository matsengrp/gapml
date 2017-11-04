import csv
import argparse
from typing import List

from barcode_events import Event
from barcode_events import BarcodeEvents
from all_reads import CellReads
from cell_state import CellTypeTree
from constants import BARCODE_V7
from constants import BARCODE_V7_LEN
from constants import CONTROL_ORGANS
from constants import NO_EVENT_STRS
from constants import NUM_BARCODE_V7_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        'reads_file', type=str, help='Collapsed reads file: format 7B')
    return parser.parse_args()


def process_event_format7B(event_str: str):
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
        return Event(event_pos, del_len, insert_str="")
    elif event_type_str == "I":
        return Event(event_pos, del_len=0, insert_str=event_split[2])
    else:
        raise ValueError("Unrecognized event: %s" % event_str)


def process_barcode_format7B(target_str_list: List[str], cell_type_tree: CellTypeTree):
    """
    Takes a barcode encoded in 7B format and returns the list of events
    """
    # Create a list of event strings for each target
    target_evt_strs = [
        target_str.split("&") for i, target_str in enumerate(target_str_list)
    ]
    # Filter the list of event strings so that we have no NO_EVENT_STRS
    target_evt_strs = [[
        evt for evt in target_evts if evt not in NO_EVENT_STRS
    ] for target_evts in target_evt_strs]
    # Create unique events
    uniq_evt_strs = list(set([
        evt_str for targ_list in target_evt_strs for evt_str in targ_list
    ]))
    events = [
        process_event_format7B(event_str) for event_str in uniq_evt_strs
    ]
    # For each target, associate list of events
    event_str_dict = {
        event_str: i for i, event_str in enumerate(uniq_evt_strs)
    }
    target_evts = [
        [event_str_dict[evt_str] for evt_str in target_evts]
        for target_evts in target_evt_strs
    ]
    # Associate each event with the matching targets
    for target_idx, evts in enumerate(target_evts):
        for evt in evts:
            events[evt].targets.append(target_idx)

    return BarcodeEvents(target_evts, events, cell_type_tree)


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

    return CellReads(all_barcodes)


def main():
    args = parse_args()
    cell_reads = parse_reads_file_format7B(args.reads_file)


if __name__ == "__main__":
    main()
