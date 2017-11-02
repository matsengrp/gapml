from Bio import pairwise2
from math import log
import warnings

class AlignerNW():
    """
    assuming not perfect sequencing (no PCR or sequencing error) we make our
    mismatch penalty effectively infinite
    """
    def __init__(self, gap_open: float = -10, gap_extend: float = -.5):
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def events(self, sequence: str, reference: str):
        if not set(sequence) <= set('ACGT'):
            raise ValueError('invalid nucleotide sequence: {}'.format(sequence))
        if not set(reference) <= set('ACGT'):
            raise ValueError('invalid nucleotide sequence: {}'.format(reference))
        alns = pairwise2.align.globalms(sequence, reference,
                                        0, -10**10, self.gap_open, self.gap_extend)
        if len(alns) > 1:
            warnings.warn('{} optimal alignments, just using first'.format(len(alns)))
        events = []
        reference_position = 0
        in_event = False
        for sequence_nucleotide, reference_nucleotide in zip(*alns[0][0:2]):
            # TODO: handle mismatches somehow, currently raise error
            if sequence_nucleotide != reference_nucleotide and not \
               (sequence_nucleotide == '-' or reference_nucleotide == '-'):
               raise NotImplementedError('mismatch {}, {}'.
                              format(sequence_nucleotide, reference_nucleotide))
            if not in_event:
                if sequence_nucleotide == '-':
                    in_event = True
                    event_start = reference_position
                    insertion = ''
            else:
                if reference_nucleotide == '-':
                    insertion += sequence_nucleotide
                    if reference_position == len(reference) - 1:
                        in_event = False
                        event_end = reference_position + 1
                        events.append((event_start, event_end, insertion))
                else:
                    in_event = False
                    event_end = reference_position
                    events.append((event_start, event_end, insertion))
            reference_position += (reference_nucleotide is not '-')
        print(events)


# TODO:  can define different affine gap functions for each sequence and can be site-aware
#
# def gap_function(x, y):  # x is gap position in seq, y is gap length
#     '''this is where we can introduce knowledge of cut sites'''
#     if y == 0:  # No gap
#         return 0
#     elif y == 1:  # Gap open penalty
#         return -2
#     return - (2 + y/4.0 + log(y)/2.0)
#
# alignment = pairwise2.align.globalmc("ACCCCCGT", "ACG", 5, -4,
#                                       gap_function, gap_function)
