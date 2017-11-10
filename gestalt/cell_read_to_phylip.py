import argparse
from typing import Dict
import numpy as np

from read_seq_data import parse_reads_file_format7B
from all_reads import CellReads


def parse_args():
    parser = argparse.ArgumentParser(description='read cell file')
    parser.add_argument(
        '--reads_file',
        type=str,
        default='/fh/fast/matsen_e/gestalt/fish_7B_UMI_collapsed_reads/fish_7B_UMI_collapsed_reads.txt',
        help='Collapsed reads file: format 7B')
    return parser.parse_args()


def make_phylip_lines(cell_reads: CellReads, evt_to_id_dict: Dict[str, int]):
    """
    convert each BarcodeEvents object to PHYLIP MIX inputs
    """
    num_events = len(cell_reads.event_str_ids)
    lines = []
    for barcode_i, barcode in enumerate(cell_reads.uniq_barcodes):
        event_idxs = [
            evt_to_id_dict[evt.get_str_id()] for evt in barcode.get_uniq_events()
        ]
        event_arr = np.zeros((num_events, ), dtype=int)
        event_arr[event_idxs] = 1
        event_encoding = "".join([str(c) for c in event_arr.tolist()])
        seq_name = str(barcode_i)
        seq_name += " " * (10 - len(seq_name))
        lines.append("%s%s\n" % (seq_name, event_encoding))
    return lines


def scale_to_phylip_weights(value: int, min_val: int, max_val: int):
    """
    Event weights to PHYLIP encoding
    """
    WEIGHT_ARRAY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    WEIGHT_ARRAY_LEN = len(WEIGHT_ARRAY)

    max_log = np.log(max_val - min_val)
    value_log = np.log(value)
    # TODO: check this calculation
    ret = int(
        np.round(((value_log - min_val) / max_log) * (WEIGHT_ARRAY_LEN - 1)))
    return WEIGHT_ARRAY[ret]


def make_phylip_weights(cell_reads: CellReads, evt_to_id_dict: Dict[str, int]):
    """
    Make PHYLIP weights for each event
    """
    num_events = len(cell_reads.event_str_ids)
    max_count = max(cell_reads.event_abundance.values())
    event_weights = [0] * num_events
    for evt_str_id, evt_abundance in cell_reads.event_abundance.items():
        evt_id = evt_to_id_dict[evt_str_id]
        event_weights[evt_id] = scale_to_phylip_weights(
            evt_abundance, 0, max_count)
    return event_weights


def convert_cell_reads_to_phylip(
        file_name: str,
        phylip_infile: str = "infile",
        phylip_weights_file: str = "weights",
):
    """
    Convert cell read file to phylip mix input
    """
    cell_reads = parse_reads_file_format7B(file_name)
    evt_to_id_dict = {
        evt_str: i
        for i, evt_str in enumerate(cell_reads.event_str_ids)
    }
    num_events = len(cell_reads.event_str_ids)

    phylip_lines = make_phylip_lines(cell_reads, evt_to_id_dict)
    with open(phylip_infile, "w") as f:
        f.write("%d %d\n" % (len(phylip_lines), num_events))
        f.writelines(phylip_lines)

    phylip_weights = make_phylip_weights(cell_reads, evt_to_id_dict)
    with open(phylip_weights_file, "w") as f:
        f.write("%s\n" % ("".join(phylip_weights)))


def main():
    args = parse_args()
    convert_cell_reads_to_phylip(args.reads_file)


if __name__ == "__main__":
    main()
