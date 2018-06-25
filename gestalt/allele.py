from typing import List, Dict, Tuple
import re
import numpy as np
from numpy.random import choice

from alignment import Aligner

from indel_sets import TargetTract
from allele_events import AlleleEvents, Event
from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS
from barcode_metadata import BarcodeMetadata
from target_status import TargetDeactTract, TargetStatus

class AlleleList:
    """
    Contains a list of alleles -- assumes we observe the same length alleles in each cell
    (and we know exactly which allele is which)
    """
    def __init__(self, allele_strs: List[List[str]], bcode_meta: BarcodeMetadata):
        self.alleles = [Allele(a, bcode_meta) for a in allele_strs]
        self.bcode_meta = bcode_meta

    def get_event_encoding(self, aligner: Aligner = None):
        """
        @return Tuple of AlleleEvents, each position in tuple for each barcode
        """
        return tuple(a.get_event_encoding(aligner) for a in self.alleles)

    def observe_with_errors(self, error_rate: float):
        """
        @return AlleleList where each allele is observed with error
        """
        alleles_err = [a.observe_with_errors(error_rate) for a in self.alleles]
        return AlleleList(
                [a.allele for a in alleles_err],
                self.bcode_meta)

    def process_events(self, events_list: List[List[Tuple[int, int, str]]]):
        """
        Given a list of observed events, reset the Allele object, rerun the events and recreate the allele
        Updates the Allele object for each barcode
        Assumes all events are NOT overlapping!!!
        """
        for a, evts in zip(self.alleles, events_list):
            a.process_events(evts)

class Allele:
    '''
    GESTALT target array with spacer sequences
    v7 allele from GESTALT paper Table S4 is unedited allele
    initial allele state equal to v7 by default

    TODO: check that we are using the right allele
    '''
    INDEL_TRANS = {'A':'-', 'C':'-','G':'-','T':'-','a':None,'c':None,'g':None,'t':None}

    def __init__(self,
                 allele: List[str],
                 bcode_meta: BarcodeMetadata):
        """
        @param allele: the current state of the allele
        @param bcode_meta: barcode metadata
        """
        # an editable copy of the allele (as a list for mutability)
        self.allele = list(allele)

        self.bcode_meta = bcode_meta

    def get_target_status(self):
        """
        @return List[int], index of the targets that can be cut, e.g. the targets where the crucial positions
        are not modified
        """
        # TODO: right now this code is pretty inefficient... but only used by simulator i think?
        events = self.get_event_encoding().events
        target_status = TargetStatus()
        for evt in events:
            min_deact, max_deact = self.bcode_meta.get_min_max_deact_targets(evt)
            target_status = target_status.add_target_tract(
                    TargetTract(min_deact, min_deact, max_deact, max_deact))
        return target_status

    def get_active_targets(self):
        targ_stat = self.get_target_status()
        return targ_stat.get_active_targets(self.bcode_meta)

    def indel(self,
              target1: int,
              target2: int,
              left_del_len: int = 0,
              right_del_len: int = 0,
              insertion: str = ''):
        '''
        a utility function for deletion/insertion

        @param target1: index of target with cut
        @param target2: index of target with cut
        if target1 != target2, create an inter-target deletion, otherwise focal deletion
        @param left_del_len: number of nucleotides to delete to the left
        @param right_del_len: number of nucleotides to delete to the right
        @param insertion: sequence placed between target deletions
        '''
        active_targets = self.get_active_targets()
        assert(target1 in active_targets)
        assert(target2 in active_targets)

        # TODO: make this code more efficient
        # indices into the self.allele list (accounting for the spacers)
        index1 = 1 + 2 * min(target1, target2)
        index2 = 1 + 2 * max(target1, target2)
        #  Determine which can cut
        cut_site = self.bcode_meta.cut_sites[target1]
        # sequence left of cut
        left = ','.join(self.allele[:index1 + 1])[:-cut_site]
        # allele sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            center = ('-' * cut_site + ',' + ','.join(
                self.allele[(index1 + 1):index2]).translate(str.maketrans(self.INDEL_TRANS)) +
                ',' + '-' * (len(self.allele[index2]) - cut_site))
        # sequence right of cut
        right = self.allele[index2][-cut_site:] + ',' +  ','.join(self.allele[index2 + 1:])
        # left delete
        deleted = 0
        for position, letter in reversed(list(enumerate(left))):
            if deleted == left_del_len:
                break
            if letter is not ',':
                left = left[:position] + '-' + left[(position + 1):]
                deleted += 1
        # right delete
        deleted = 0
        for position, letter in enumerate(right):
            if deleted == right_del_len:
                break
            if letter is not ',':
                right = right[:position] + '-' + right[(position + 1):]
                deleted += 1
        # put it back together
        self.allele = (left + insertion + center + right).split(',')

    def get_events(self, aligner: Aligner = None, left_align: bool = False):
        '''
        @param aligner: Aligner object
                        must have events() method
                        None returns the observable events using the true allele state
        @param left_align: left-align indels
        return the list of observable indel events in the allele
        '''
        if aligner is None:
            # find the indels
            events = []
            insertion_total = 0
            for indel in re.compile('[-acgt]+').finditer(str(self)):
                start = indel.start() - insertion_total
                # find the insertions(s) in this indel
                insertion = ''.join(
                    insertion.group(0)
                    for insertion in re.compile('[acgt]+').finditer(indel.group(0))
                )
                insertion_total += len(insertion)
                end = indel.end() - insertion_total
                events.append((start, end, insertion))
        else:
            sequence = str(self).replace('-', '').upper()
            reference = ''.join(self.bcode_meta.unedited_barcode).upper()
            events = aligner.events(sequence, reference)
        if left_align:
            raise NotImplementedError()
            # for event in events:
            #     allele_copy =
        return events

    def get_event_encoding(self, aligner: Aligner = None):
        """
        @return a AlleleEvents version of this allele
        """
        raw_events = self.get_events(aligner=aligner)
        events = []
        for evt_i, evt in enumerate(raw_events):
            matching_targets = []
            for tgt_i, cut_site in enumerate(self.bcode_meta.abs_cut_sites):
                if evt[0] <= cut_site and evt[1] >= cut_site:
                    matching_targets.append(tgt_i)

            events.append(Event(
                evt[0],
                evt[1] - evt[0],
                min_target=min(matching_targets) if matching_targets else None,
                max_target=max(matching_targets) if matching_targets else None,
                insert_str=evt[2],
            ))

        return AlleleEvents(events)

    def process_events(self, events: List[Tuple[int, int, str]]):
        """
        Given a list of observed events, resets the allele, rerun the events, and recreate the allele
        Assumes all events are NOT overlapping!!!
        """
        # initialize allele to unedited states
        self.allele = list(self.bcode_meta.unedited_barcode)
        sub_str_lens = [len(sub_str) for sub_str in self.allele]
        total_len = sum(sub_str_lens)
        for evt in events:
            del_start = evt[0]
            del_end = evt[1]
            insertion_str = evt[2]

            # special case for insertion off the 3' end
            if del_start == del_end == total_len:
                self.allele[-1] += insertion_str
                continue

            # Determine which substrings to start and end at
            # TODO: make this more efficient?
            idx = 0
            for sub_str_idx, sub_str in enumerate(self.allele):
                sub_str_len = sub_str_lens[sub_str_idx]
                if idx + sub_str_len >= del_start and idx <= del_start:
                    substr_start = sub_str_idx
                    substr_start_inner_idx = del_start - idx
                if idx + sub_str_len >= del_end and idx <= del_end:
                    substr_end = sub_str_idx
                    substr_end_inner_idx = del_end - idx
                idx += sub_str_len

            # Now do the actual deletions
            for sub_str_idx in range(substr_start, substr_end + 1):
                curr_substr = self.allele[sub_str_idx]
                start_idx = substr_start_inner_idx if sub_str_idx == substr_start else 0
                end_idx = substr_end_inner_idx if sub_str_idx == substr_end else sub_str_lens[
                    sub_str_idx]

                new_sub_str = []
                non_insert_idx = 0
                deleting = False
                for substr_char in curr_substr:
                    if non_insert_idx == start_idx:
                        deleting = True
                        # Do the insertion at the start of the deletion in the first substring block
                        if sub_str_idx == substr_start:
                            new_sub_str.append(insertion_str)
                    if non_insert_idx == end_idx:
                        deleting = False
                    if deleting:
                        new_sub_str.append("-")
                    else:
                        new_sub_str.append(substr_char)
                    if substr_char in "ACTG-":
                        non_insert_idx += 1
                self.allele[sub_str_idx] = "".join(new_sub_str)

    def observe_with_errors(self, error_rate: float):
        """
        @param error_rate: probability of each base being erroneous
        @return copy of allele with random errors, uniform over alternative
                bases
        NOTE: to be used after any editing, since get_active_targets will not
              work as expected with base errors
        """
        assert (0 <= error_rate <= 1)
        if error_rate == 0:
            return self
        allele_with_errors = []
        nucs = 'acgt'
        for substr_idx, substr in enumerate(self.allele):
            new_substr = ''
            for substr_char in substr:
                if substr_char == '-':
                    new_substr += substr_char
                else:
                    probs = [(error_rate/3 if nuc != substr_char.lower()
                                          else (1 - error_rate))
                             for nuc in nucs]
                    new_substr += choice(list(nucs.upper() if substr_char.isupper() else nucs),
                                         p=probs)
            allele_with_errors.append(new_substr)

        return Allele(allele=allele_with_errors,
                       bcode_meta=self.bcode_meta)


    def __repr__(self):
        return ''.join(self.allele)
