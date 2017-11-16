from typing import List
from typing import Dict
import numpy as np

from cell_state import CellTypeTree
from constants import NUM_BARCODE_V7_TARGETS

"""
Objects for representing a barcode using an event-encoded format
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
        @param insert_str: sequence of nucleotides inserted, if *, then it is wildcard
        @param targets: which targets this event is associated with
        """
        return tuple.__new__(cls, (start_pos, del_len, min_target, max_target, insert_str))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.min_target, self.max_target, self.insert_str)

    @property
    def is_wildcard(self):
        return self.insert_str == "*"

    @property
    def start_pos(self):
        return self[0]

    @property
    def del_len(self):
        return self[1]

    @property
    def del_end(self):
        return self.start_pos + self.del_len - 1

    @property
    def min_target(self):
        return self[2]

    @property
    def max_target(self):
        return self[3]

    @property
    def insert_str(self):
        return self[4]

    @property
    def start_end(self):
        return (self.min_target, self.max_target)

    def generalized_equals(self, child_evt):
        if (self.start_pos == child_evt.start_pos and
                self.del_len == child_evt.del_len and
                self.min_target == child_evt.min_target and
                self.max_target == child_evt.max_target):
            if self.is_wildcard or child_evt.is_wildcard:
                return True
            else:
                return self.insert_str == child_evt.insert_str
        return False

class BarcodeEvents:
    """
    Represents a barcode in event-encoding.
    Efficient for estimation procedures.

    Use this representation for cleaned barcode representation where each target
    can be associated with at most a single event.
    """
    def __init__(self, events: List[Event] = [], num_targets=10):
        """
        @param events: tuples of tuples of events
                    a tuple of events means either event may have happened
        """
        self.events = events
        start_ends = [[evt.start_pos, evt.del_end] for evt in events]
        start_ends = [i for tup in start_ends for i in tup]
        for i in range(len(start_ends) - 1):
            assert(start_ends[i] <= start_ends[i + 1])
        self.num_targets = num_targets

    def __str__(self):
        if self.events:
            return "..".join([str(evts) for evts in self.events])
        else:
            return "[]"
