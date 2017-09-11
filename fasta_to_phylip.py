from __future__ import print_function
import argparse

from Bio import SeqIO
import numpy as np

def create_event_ids(observed_barcode):
    barcode_len = len(observed_barcode)
    events = []
    end_idx = 0
    while end_idx < barcode_len:
        find_start_idx = observed_barcode[end_idx:].find("-")
        if find_start_idx > 0:
            start_idx = find_start_idx + end_idx
            end_idx = start_idx
            while observed_barcode[end_idx] == "-":
                end_idx += 1
            events.append((start_idx, end_idx))
        else:
            break
    return events

def main():
    parser = argparse.ArgumentParser(description='convert to MIX')
    parser.add_argument('fasta', type=str, help='fasta input')
    parser.add_argument('--out', type=str, help='output to feed to phylip mix')
    args = parser.parse_args()

    # Naive processing of events
    all_events = set()
    processed_seqs = []
    for record in SeqIO.parse(args.fasta, "fasta"):
        seq_events = create_event_ids(record.seq)
        processed_seqs.append(seq_events)
        all_events.update(seq_events)
    all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}
    num_events = len(all_event_dict)

    # Output file for PHYLIP
    lines = []
    for seq_events in processed_seqs:
        event_idxs = [all_event_dict[seq_ev] for seq_ev in seq_events]
        event_arr = np.zeros((num_events,), dtype=int)
        event_arr[event_idxs] = 1
        lines.append("".join([str(c) for c in event_arr.tolist()]) + "\n")

    with open(args.out, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    main()
