import csv
import argparse
from typing import List

from models import Event
from models import InsertionEvent
from models import DeletionEvent
from models import BarcodeEvents
from models import CellReads
from constants import BARCODE_V7
from constants import BARCODE_V7_LEN
from constants import CONTROL_ORGANS
from constants import NO_EVENT_STRS
from constants import NUM_BARCODE_V7_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument('reads_file', type=str, help='Collapsed reads file: format 7B')
    return parser.parse_args()


def process_event_format7B(event_str: str):
    """
    Takes a single event string and creates an event object
    """
    event_split = event_str.split("+")
    event_type_str = event_split[0][-1]
    event_len = int(event_split[0][:-1])
    event_pos = int(event_split[1])
    if event_type_str == "D":
        return DeletionEvent(event_len, event_pos)
    elif event_type_str == "I":
        return InsertionEvent(event_len, event_pos, event_split[2])
    else:
        raise ValueError("Unrecognized event: %s" % event_str)


def process_barcode_format7B(target_str_list: List[str], organ: str):
    """
    Takes a barcode encoded in 7B format and returns the list of events
    """
    target_evt_strs = [target_str.split("&") for i, target_str in enumerate(target_str_list)]
    barcode_evt_strs = [evt for target_evts in target_evt_strs for evt in target_evts if evt not in NO_EVENT_STRS]
    barcode_evt_strs = set(barcode_evt_strs)
    events = [process_event_format7B(event_str) for event_str in barcode_evt_strs]
    return BarcodeEvents(events, organ)


def parse_reads_file_format7B(file_name, target_hdr_fmt="target%d", max_read=None):
    """
    @param max_read: maximum number of barcodes to read (for debugging purposes)
    """
    all_barcodes = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        target_start_idx = header.index(target_hdr_fmt % 1)
        for row in reader:
            organ = row[0]
            if organ not in CONTROL_ORGANS:
                barcode_events = process_barcode_format7B(
                    [row[target_start_idx + i] for i in range(NUM_BARCODE_V7_TARGETS)],
                    organ,
                )
                all_barcodes.append(barcode_events)
                if max_read is not None and len(all_barcodes) == max_read:
                    break

    return CellReads(all_barcodes)


def main():
    args = parse_args()
    cell_reads = parse_reads_file_format7B(args.reads_file)


if __name__ == "__main__":
    main()
