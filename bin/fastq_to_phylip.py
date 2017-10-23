"""
Converts the fastq files from simulation.py to input to PHYLIP MIX
Makes the binary file format
"""

from __future__ import print_function
import argparse

from Bio import SeqIO
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='convert to MIX')
    parser.add_argument('fastq', type=str, help='fastq input')
    parser.add_argument('--out', type=str, help='output to feed to phylip mix')
    args = parser.parse_args()

    # Naive processing of events
    all_events = set()
    processed_seqs = []
    for record in SeqIO.parse(args.fastq, 'fastq'):
        seq_events = record.description.split(',')
        processed_seqs.append((record.id, seq_events))
        all_events.update(seq_events)
    all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}
    num_events = len(all_event_dict)

    print(all_events)

    # Output file for PHYLIP
    # Format is very dumb: species name must be 10 characters long, followed by sequence of 0 and 1s
    lines = []
    for seq_name, seq_events in processed_seqs:
        event_idxs = [all_event_dict[seq_ev] for seq_ev in seq_events]
        event_arr = np.zeros((num_events,), dtype=int)
        event_arr[event_idxs] = 1
        event_encoding = "".join([str(c) for c in event_arr.tolist()])
        seq_name += " " * (10 - len(seq_name))
        lines.append("%s%s\n" % (seq_name, event_encoding))

    with open(args.out, "w") as f:
        f.write("%d %d\n" % (len(lines), num_events))
        f.writelines(lines)

if __name__ == "__main__":
    main()
