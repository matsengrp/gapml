from typing import List, Dict, Tuple
import re
import numpy as np
from numpy.random import choice

from alignment import Aligner

from barcode_events import BarcodeEvents, Event
from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS


class Barcode:
    '''
    GESTALT target array with spacer sequences
    v7 barcode from GESTALT paper Table S4 is unedited barcode
    initial barcode state equal to v7 by default

    TODO: check that we are using the right barcode
    '''
    INDEL_TRANS = {'A':'-', 'C':'-','G':'-','T':'-','a':None,'c':None,'g':None,'t':None}

    def __init__(self,
                 barcode: List[str] = BARCODE_V7,
                 unedited_barcode: List[str] = BARCODE_V7,
                 cut_sites: List[int] = [6] * NUM_BARCODE_V7_TARGETS):
        """
        @param barcode: the current state of the barcode
        @param unedited_barcode: the original state of the barcode
        @param cut_sites: offset from 3' end of target for Cas9 cutting,
                        so a cut_site of 6 means that we start inserting
                        such that the inserted seq is 6 nucleotides from
                        the 3' end of the target
        """
        # The original barcode
        self.unedited_barcode = unedited_barcode
        self.orig_substr_lens = [len(s) for s in unedited_barcode]
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(barcode)
        self.cut_sites = cut_sites
        # number of targets
        self.n_targets = (len(self.barcode) - 1) // 2
        # a list of target indices that have a DSB and need repair
        self.needs_repair = set()
        # absolute positions of cut locations
        self.abs_cut_sites = [
            sum(self.orig_substr_lens[:2 * (i + 1)]) - cut_sites[i] for i in range(self.n_targets)
        ]
        assert (self.n_targets == len(self.cut_sites))

    def get_active_targets(self):
        """
        @return the index of the targets that can be cut, e.g. the targets that have no DSBs and are unmodified
        """
        # TODO: right now this code is pretty inefficient. we might want to cache which targets are active
        matches = [
            i not in self.needs_repair and self.unedited_barcode[2 * i + 1] == self.barcode[2 * i + 1]
            for i in range(self.n_targets)
        ]
        return np.where(matches)[0]

    def cut(self, target_idx):
        """
        Marks this target as having a DSB
        """
        assert(target_idx in self.get_active_targets())
        self.needs_repair.add(target_idx)

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
        assert(target1 in self.needs_repair)
        assert(target2 in self.needs_repair)

        # TODO: make this code more efficient
        # indices into the self.barcode list (accounting for the spacers)
        index1 = 1 + 2 * min(target1, target2)
        index2 = 1 + 2 * max(target1, target2)
        #  Determine which can cut
        cut_site = self.cut_sites[target1]
        # sequence left of cut
        left = ','.join(self.barcode[:index1 + 1])[:-cut_site]
        # barcode sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            center = ('-' * cut_site + ',' + ','.join(
                self.barcode[(index1 + 1):index2]).translate(str.maketrans(self.INDEL_TRANS)) +
                ',' + '-' * (len(self.barcode[index2]) - cut_site))
        # sequence right of cut
        right = self.barcode[index2][-cut_site:] + ',' +  ','.join(self.barcode[index2 + 1:])
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
        self.barcode = (left + insertion + center + right).split(',')
        # Update needs_repair
        # Currently assumes that during inter-target indels, any cuts in the middle
        # also get repaired.
        self.needs_repair = self.needs_repair.difference(set(range(target1, target2 + 1)))

    def get_events(self, aligner: Aligner = None, left_align: bool = False):
        '''
        @param aligner: Aligner object
                        must have events() method
                        None returns the observable events using the true barcode state
        @param left_align: left-align indels
        return the list of observable indel events in the barcode
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
            reference = ''.join(self.unedited_barcode).upper()
            events = aligner.events(sequence, reference)
        if left_align:
            raise NotImplementedError()
            # for event in events:
            #     barcode_copy =
        return events

    def get_event_encoding(self):
        """
        @return a BarcodeEvents version of this barcode
        """
        raw_events = self.get_events()
        target_evts = [None for i in range(self.n_targets)]
        events = []
        for evt_i, evt in enumerate(raw_events):
            matching_targets = [
                target_idx for target_idx, cut_site in enumerate(self.abs_cut_sites)
                if evt[0] <= cut_site and evt[1] >= cut_site
            ]
            assert(matching_targets)

            for t in matching_targets:
                assert(target_evts[t] is None)
                target_evts[t] = evt_i

            events.append(Event(
                evt[0],
                evt[1] - evt[0],
                min_target=min(matching_targets),
                max_target=max(matching_targets),
                insert_str=evt[2],
            ))

        return BarcodeEvents(events)

    def process_events(self, events: List[Tuple[int, int, str]]):
        """
        Given a list of observed events, rerun the events and recreate the barcode
        Assumes all events are NOT overlapping!!!
        """
        sub_str_lens = [len(sub_str) for sub_str in self.barcode]
        total_len = sum(sub_str_lens)
        for evt in events:
            del_start = evt[0]
            del_end = evt[1]
            insertion_str = evt[2]

            # special case for insertion off the 3' end
            if del_start == del_end == total_len:
                self.barcode[-1] += insertion_str
                continue

            # Determine which substrings to start and end at
            # TODO: make this more efficient?
            idx = 0
            for sub_str_idx, sub_str in enumerate(self.barcode):
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
                curr_substr = self.barcode[sub_str_idx]
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
                self.barcode[sub_str_idx] = "".join(new_sub_str)

    def observe_with_errors(self, error_rate: float):
        """
        @param error_rate: probability of each base being erroneous
        @return copy of barcode with random errors, uniform over alternative
                bases
        NOTE: to be used after any editing, since get_active_targets will not
              work as expected with base errors
        """
        assert (0 <= error_rate <= 1)
        if error_rate == 0:
            return self
        barcode_with_errors = []
        nucs = 'acgt'
        for substr_idx, substr in enumerate(self.barcode):
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
            barcode_with_errors.append(new_substr)

        return Barcode(barcode=barcode_with_errors,
                       unedited_barcode=self.unedited_barcode,
                       cut_sites=self.cut_sites)


    def __repr__(self):
        return ''.join(self.barcode)
