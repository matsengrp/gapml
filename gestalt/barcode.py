from typing import List, Dict, Tuple
import re

import numpy as np

from barcode_events import BarcodeEvents, Event
from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS


class Barcode:
    '''
    GESTALT target array with spacer sequences
    v7 barcode from GESTALT paper Table S4 is unedited barcode
    initial barcode state equal to v7 by default

    TODO: check that we are using the right barcode
    '''

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
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(barcode)
        self.sub_str_lens = [len(sub_str) for sub_str in barcode]
        self.total_len = sum(self.sub_str_lens)
        self.cut_sites = cut_sites
        # number of targets
        self.n_targets = (len(self.barcode) - 1) // 2
        # a list of target indices that have a DSB and need repair
        self.needs_repair = set()
        # absolute positions of cut locations
        self.abs_cut_sites = [
            sum(self.sub_str_lens[:2 * (i + 1)]) - cut_sites[i] for i in range(self.n_targets)
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
        #  Deletermine which can cut
        cut_site = self.cut_sites[target1]
        # sequence left of cut
        left = ','.join(self.barcode[:index1 + 1])[:-cut_site]
        # barcode sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            maketrans = str.maketrans
            center = '-' * cut_site + ',' + ','.join(
                self.barcode[(index1 + 1):index2]).translate(
                    maketrans('ACGTacgt', '-' * 8)) + ',' + '-' * (
                        len(self.barcode[index2]) - cut_site)
        # sequence right of cut
        right = ','.join(
            self.barcode[index2:])[len(self.barcode[index2]) - cut_site:]
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

    def get_events(self):
        '''return the list of observable indel events in the barcdoe'''
        events = []
        # find the indels
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
        return events

    def get_event_encoding(self):
        raw_events = self.get_events()
        target_evts = [[] for i in range(self.n_targets)]
        events = []
        for evt_i, evt in enumerate(raw_events):
            matching_targets = [
                target_idx for target_idx, cut_site in enumerate(self.abs_cut_sites)
                if evt[0] <= cut_site and evt[1] >= cut_site
            ]
            for t in matching_targets:
                target_evts[t].append(evt_i)
            events.append(Event(
                start_pos = evt[0],
                del_len = evt[1] - evt[0],
                insert_str = evt[2],
                targets = matching_targets,
            ))

        # TODO: add organ in here
        return BarcodeEvents(target_evts, events, None)

    def process_events(self, events: List[Tuple[int, int, str]]):
        """
        Given a list of observed events, rerun the events and recreate the barcode
        Assumes all events are NOT overlapping!!!
        """
        for evt in events:
            del_start = evt[0]
            del_end = evt[1]
            insertion_str = evt[2]

            # special case for insertion off the 3' end
            if del_start == del_end == self.total_len:
                self.barcode[-1] += insertion_str
                continue

            # Determine which substrings to start and end at
            # TODO: make this more efficient?
            idx = 0
            for sub_str_idx, sub_str in enumerate(self.barcode):
                sub_str_len = self.sub_str_lens[sub_str_idx]
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
                end_idx = substr_end_inner_idx if sub_str_idx == substr_end else self.sub_str_lens[
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
                    if substr_char in "ACTG":
                        non_insert_idx += 1
                self.barcode[sub_str_idx] = "".join(new_sub_str)


    def __repr__(self):
        return str(''.join(self.barcode))
