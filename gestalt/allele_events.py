from typing import List
from typing import Dict
import numpy as np

from barcode_metadata import BarcodeMetadata
from cell_state import CellTypeTree
from constants import NUM_BARCODE_V7_TARGETS, NO_EVT_STR

"""
Objects for representing a allele using an event-encoded format
"""
class Event(tuple):
    def __new__(
        cls,
        start_pos: int,
        del_len: int,
        min_target: int,
        max_target: int,
        insert_str: str = ""):
        """
        @param start_pos: position where event begins
        @param del_len: number of nucleotides deleted
        @param min_target: min target that got cut
        @param max_target: max target that got cut
        @param insert_str: sequence of nucleotides inserted
        """
        assert start_pos >= 0
        assert min_target >= 0
        assert min_target <= max_target
        assert del_len >= 0
        return tuple.__new__(cls, (start_pos, del_len, min_target, max_target, insert_str))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.min_target, self.max_target, self.insert_str)

    @property
    def start_pos(self):
        return self[0]

    @property
    def del_len(self):
        return self[1]

    @property
    def del_end(self):
        return self.start_pos + self.del_len

    @property
    def min_target(self):
        return self[2]

    @property
    def max_target(self):
        return self[3]

    @property
    def can_hide_cuts(self):
        return self.min_target + 2 <= self.max_target

    @property
    def insert_str(self):
        return self[4]

    @property
    def insert_len(self):
        return len(self[4])

    @property
    def start_end(self):
        return (self.min_target, self.max_target)

    def hides(self, other):
        """return True if this event hides another event"""
        if self.min_target < other.min_target and other.max_target < self.max_target:
            assert other.start_pos >= self.start_pos and other.del_end <= self.del_end
            return True
        else:
            return False

    def get_trim_lens(self, bcode_meta: BarcodeMetadata):
        """
        @return left trim length, right trim length
        """
        left_trim_len = bcode_meta.abs_cut_sites[self.min_target] - self.start_pos
        right_trim_len = self.del_end - bcode_meta.abs_cut_sites[self.max_target]
        assert left_trim_len >= 0
        assert right_trim_len >= 0
        assert left_trim_len <= bcode_meta.left_max_trim[self.min_target]
        assert right_trim_len <= bcode_meta.right_max_trim[self.max_target]
        return left_trim_len, right_trim_len

    def get_min_max_deact_targets(self, bcode_meta: BarcodeMetadata):
        """
        @return minimum deactivated target, maximum deactivated target
        """
        if self.min_target > 0 and self.start_pos <= bcode_meta.pos_sites[self.min_target - 1][1]:
            min_deact_target = self.min_target - 1
        else:
            min_deact_target = self.min_target

        if self.max_target < bcode_meta.n_targets - 1 and bcode_meta.pos_sites[self.max_target + 1][0] < self.del_end:
            max_deact_target = self.max_target + 1
        else:
            max_deact_target = self.max_target

        assert min_deact_target is not None
        assert max_deact_target is not None
        return min_deact_target, max_deact_target

class AlleleEvents:
    """
    Represents a allele in event-encoding.
    Efficient for estimation procedures.

    Use this representation for cleaned allele representation where each target
    can be associated with at most a single event.
    """
    def __init__(self, events: List[Event] = [], num_targets=10):
        """
        @param events: tuples of tuples of events
                    a tuple of events means either event may have happened
        """
        self.events = sorted(events, key=lambda evt: evt.start_pos)
        start_ends = [[evt.start_pos, evt.del_end] for evt in self.events]
        start_ends = [i for tup in start_ends for i in tup]
        for i in range(len(start_ends) - 1):
            assert(start_ends[i] <= start_ends[i + 1])
        for i in range(len(self.events) - 1):
            if (self.events[i].max_target >= self.events[i + 1].min_target):
                print(events)
            assert(self.events[i].max_target < self.events[i + 1].min_target)
        self.num_targets = num_targets

    def __lt__(self, other):
        # Needed for robinson fould calculation for some reason
        return str(self) < str(other)

    def __le__(self, other):
        # Needed for robinson fould calculation for some reason
        return str(self) <= str(other)

    def __eq__(self, other):
        return self.events == other.events

    def __hash__(self):
        return hash(tuple(self.events))

    def __str__(self):
        if self.events:
            return "=".join(["_".join([str(e) for e in evts]) for evts in self.events])
        else:
            return NO_EVT_STR
