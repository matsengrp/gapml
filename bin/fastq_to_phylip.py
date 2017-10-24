"""
Converts the fastq files from simulation.py to input to PHYLIP MIX
Makes the binary file format
"""

from __future__ import print_function
import argparse
import re
from Bio import SeqIO
import numpy as np
import warnings


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
        if str(record.seq) not in processed_seqs:
            all_events.update(seq_events)
            # list abundance and indel events
            processed_seqs[str(record.seq)] = [1, seq_events]
        else:
            processed_seqs[str(record.seq)][0] += 1
            if processed_seqs[str(record.seq)][1] != seq_events:
                warnings.warn(
                    'identical sequences have different event calls: {}, {}\nsequence: {}'
                    .format(processed_seqs[str(record.seq)][1], seq_events,
                            str(record.seq)))
    all_event_dict = {event_id: i for i, event_id in enumerate(all_events)}
    num_events = len(all_event_dict)

    # Output file for PHYLIP
    # Format is very dumb: species name must be 10 characters long, followed by sequence of 0 and 1s
    with open(args.outbase + '.phy',
              "w") as f1, open(args.outbase + '.abundance', "w") as f2:
        f1.write("%d %d\n" % (len(processed_seqs), num_events))
        f2.write('id\tabundance\n')
        for seq_id, seq in enumerate(processed_seqs):
            seq_abundance, seq_events = processed_seqs[seq]
            event_idxs = [all_event_dict[seq_ev] for seq_ev in seq_events]
            event_arr = np.zeros((num_events, ), dtype=int)
            event_arr[event_idxs] = 1
            event_encoding = "".join([str(c) for c in event_arr.tolist()])
            seq_name = 'seq{}'.format(seq_id)
            seq_name += " " * (10 - len(seq_name))
            f1.write("%s%s\n" % (seq_name, event_encoding))
            f2.write('{}\t{}\n'.format(seq_name, seq_abundance))


if __name__ == "__main__":
    main()
