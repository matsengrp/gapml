from Bio import pairwise2
from math import log
import warnings

class Aligner():
    """
    base class for aligners
    """
    def events(self, sequence: str, reference: str):
        """
        @param sequence: observed (edited) allele nucleotide sequence
        @param reference: design alleled sequence to align to
        @return events as a list of tuples of start, end, and insertion sequence
        """
        raise NotImplementedError()

class AlignerNW(Aligner):
    """
    Needleman-Wunsch alignment to identify indel events
    assuming perfect sequencing (no PCR or sequencing error) we make our
    mismatch penalty effectively infinite
    """
    def __init__(self, match: float = 0, mismatch: float = -1,
                 gap_open: float = -10, gap_extend: float = -.5, return_all=False):
        """
        @param match: match score
        @param mismatch: mismatch penalty (high default assumes only indels possible)
        @param gap_open: gap open penalty
        @param gap_extend: gap extension penalty
        @param return_all: return all equally optimal alignments, else return first
        """
        self.match = match
        self.mismatch = mismatch
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.return_all = return_all

    def events(self, sequence: str, reference: str):
        """
        @return indel events
        """
        if not set(sequence) <= set('ACGT'):
            raise ValueError('invalid nucleotide sequence: {}'.format(sequence))
        if not set(reference) <= set('ACGT'):
            raise ValueError('invalid nucleotide sequence: {}'.format(reference))
        # this function produces Needleman-Wunsch alignments
        alns = pairwise2.align.globalms(sequence, reference,
                                        self.match, self.mismatch, self.gap_open, self.gap_extend)
        if self.return_all:
            events_list = []
        for aln in alns:
            events = []
            reference_position = 0
            in_event = False
            # taking the first alignment only
            # iterate through alignment character by character
            for sequence_nucleotide, reference_nucleotide in zip(*aln[0:2]):
                # if we are not in an event, and we find dashes, we must have just
                # entered an event
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
                # if we are in an event, we can distinguish insertions from
                # deletions by which sequence has a dash
                else:
                    if reference_nucleotide == '-':
                        insertion += sequence_nucleotide
                    elif sequence_nucleotide == '-':
                        event_end += 1
                    else:
                        in_event = False
                        events.append((event_start, event_end, insertion.lower()))
                if reference_nucleotide is not '-':
                    reference_position += 1
            # special case of event ending at the end of the reference
            if in_event:
                in_event = False
                events.append((event_start, event_end, insertion.lower()))
            assert reference_position == len(reference)
            assert in_event == False

            if self.return_all:
                events_list.append(events)
            else:
                return events
        return events_list

