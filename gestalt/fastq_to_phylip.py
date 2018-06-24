"""
Converts the fastq files from simulation.py to input to PHYLIP MIX
Makes the binary file format
"""

from __future__ import print_function
from typing import Dict, List
import argparse
import re
from Bio import SeqIO
import numpy as np
import warnings
from allele_events import Event
from cell_state import CellState


def write_seqs_to_phy(processed_seqs: Dict[str, List],
                      event_dicts: List[Dict[Event, int]],
                      phy_file: str,
                      abundance_file: str,
                      encode_hidden: bool =True,
                    # TODO: currently ignored. include cell state in the future?
                    use_cell_state: bool =False):
    """
    @param processed_seqs: dict key = sequence id, dict val = [abundance, list of events]
    @param event_dicts: list for dicts, one for each barcode, key = event, dict val = event phylip id
    @param phy_file: name of file to input to phylip
    @param abundance_file: name of file with abundance values
    @param mark_hidden: whether or not to encode hidden states as "?"
    """
    num_events = sum([len(evt_dict) for evt_dict in event_dicts])

    # Some events hide others, so we build a dictionary mapping event ids to the
    # ids of the events they hide. We will use this to encode indeterminate states.
    if encode_hidden:
        hidden_events_dict = {}
        for evt_dict in event_dicts:
            for evt, evt_idx in evt_dict.items():
                hidden_evts = [evt2_idx for evt2, evt2_idx in evt_dict.items() if evt2 is not evt and evt.hides(evt2)]
                hidden_events_dict[evt_idx] = hidden_evts

    # Output file for PHYLIP
    # species name must be 10 characters long, followed by a sequence of 0s and
    # 1s indicating unique event absence and presence, respectively, with the
    # "?" character for undetermined states (i.e. hidden events)
    with open(phy_file, "w") as f1, open(abundance_file, "w") as f2:
        f1.write("%d %d\n" % (len(processed_seqs), num_events))
        f2.write('id\tabundance\n')
        for seq_id, seq_data in processed_seqs.items():
            seq_abundance = seq_data[0]
            all_seq_events = seq_data[1]
            event_idxs = [event_dicts[bcode_idx][evt] for bcode_idx, bcode_evts in enumerate(all_seq_events) for evt in bcode_evts]
            event_arr = np.array(['0' for _ in range(num_events)])
            event_arr[event_idxs] = '1'
            if encode_hidden:
                indeterminate_idxs = set([
                    hidden_idx
                    for evt_idx in event_idxs
                    for hidden_idx in hidden_events_dict[evt_idx]])
                assert(indeterminate_idxs.isdisjoint(set(event_idxs)))
                event_arr[list(indeterminate_idxs)] = "?"
            event_encoding = "".join(event_arr)
            seq_name = seq_id
            seq_name += " " * (10 - len(seq_name))
            f1.write("%s%s\n" % (seq_name, event_encoding))
            f2.write('{}\t{}\n'.format(seq_name, seq_abundance))


def main():
    parser = argparse.ArgumentParser(description='convert to MIX')
    parser.add_argument('fastq', type=str, help='fastq input')
    parser.add_argument(
        '--outbase',
        type=str,
        help='output basename for phylip file and abundance weight file')
    args = parser.parse_args()

    # Naive processing of events
    all_events = set()
    processed_seqs = {}
    for record in SeqIO.parse(args.fastq, 'fastq'):
        seq_events = [
            event.group(0)
            for event in re.compile('[0-9]*:[0-9]*:[acgt]*').finditer(
                record.description)
        ]
        record_name = "seq" + record_name
        if record_name not in processed_seqs:
            all_events.update(seq_events)
            # list abundance and indel events
            processed_seqs[record_name] = [1, seq_events]
        else:
            processed_seqs[record_name][0] += 1
            if processed_seqs[record_name][1] != seq_events:
                warnings.warn(
                    'identical sequences have different event calls: {}, {}\nsequence: {}'
                    .format(processed_seqs[record_name][1], seq_events,
                            record_name))
    all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}

    phy_file = args.outbase + '.phy',
    abundance_file = args.outbase + '.abundance'
    write_seqs_to_phy(processed_seqs, all_event_dict, phy_file, abundance_file)


if __name__ == "__main__":
    main()
