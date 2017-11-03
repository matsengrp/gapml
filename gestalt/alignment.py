from Bio import pairwise2
from math import log
import warnings

class Aligner():
    """
    base class for aligners
    """
    def events(self, sequence: str, reference: str):
        """
        @param sequence: observed (edited) barcode nucleotide sequence
        @param reference: design barcoded sequence to align to
        """
        raise NotImplementedError()

class AlignerNW(Aligner):
    """
    Needleman-Wunsch alignment to identify indel events
    assuming perfect sequencing (no PCR or sequencing error) we make our
    mismatch penalty effectively infinite
    """
    def __init__(self, gap_open: float = -10, gap_extend: float = -.5):
        """
        @param gap_open: gap open penalty
        @param gap_extend: gap extension penalty
        """
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def events(self, sequence: str, reference: str):
        """
        @return indel events
        """
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
                if sequence_nucleotide == '-' or reference_nucleotide == '-':
                    in_event = True
                    event_start = reference_position
                    event_end = event_start
                    if sequence_nucleotide == '-':
                        event_end += 1
                        insertion = ''
                    else:
                        insertion = sequence_nucleotide
            else:
                if reference_nucleotide == '-':
                    insertion += sequence_nucleotide
                elif sequence_nucleotide == '-':
                    event_end += 1
                else:
                    in_event = False
                    events.append((event_start, event_end, insertion.lower()))
            reference_position += (reference_nucleotide is not '-')
        if in_event:
            in_event = False
            events.append((event_start, event_end, insertion.lower()))
        assert reference_position == len(reference)
        assert in_event == False

        return events


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
