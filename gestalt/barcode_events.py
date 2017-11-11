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
        insert_str: str,
        min_target: int,
        max_target: int):
        """
        @param start_pos: position where event begins
        @param del_len: number of nucleotides deleted
        @param insert_str: sequence of nucleotides inserted
        @param targets: which targets this event is associated with
        """
        return tuple.__new__(cls, (start_pos, del_len, insert_str, min_target, max_target))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.insert_str, self.min_target, self.max_target)

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
    def insert_str(self):
        return self[2]

    @property
    def min_target(self):
        return self[3]

    @property
    def max_target(self):
        return self[4]

    @property
    def start_end(self):
        return (self.min_target, self.max_target)

class EventWildcard(tuple):
    def __new__(
        cls,
        start_pos: int,
        del_len: int,
        min_target: int,
        max_target: int):
        """
        @param start_pos: position where event begins
        @param del_len: number of nucleotides deleted
        """
        return tuple.__new__(cls, (start_pos, del_len, min_target, max_target))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.min_target, self.max_target)

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
        return self[-2]

    @property
    def max_target(self):
        return self[-1]

    @property
    def start_end(self):
        return (self.min_target, self.max_target)

    def is_compatible(self, evt: Event):
        if evt.start_pos == self.start_pos:
            if evt.max_target < self.max_target:
                return True
            elif evt.del_end == self.del_end:
                return True
        elif evt.del_end == self.del_end:
            if evt.min_target < self.min_target:
                return True
        return False

class UnresolvedEvents:
    def __init__(self, event: Event = None, wildcard: EventWildcard = None):
        self.event = event if event else None
        self.wildcard = wildcard if wildcard else None
        self._val = event if event else wildcard
        assert(not(self.event and self.wildcard))

    def __str__(self):
        return str(self._val)

    @property
    def start_pos(self):
        return self._val.start_pos

    @property
    def del_end(self):
        return self._val.del_end

    @property
    def start_end(self):
        return self._val.start_end

    @property
    def min_target(self):
        return self._val.min_target

    @property
    def max_target(self):
        return self._val.max_target

class BarcodeEvents:
    """
    Represents a barcode in event-encoding.
    Efficient for estimation procedures.

    Use this representation for cleaned barcode representation where each target
    can be associated with at most a single event.
    """
    ACTIVE = 0
    INACTIVE = 1
    UNRESOLVED = 2

    def __init__(self, events: List[UnresolvedEvents]):
        """
        @param events: tuples of tuples of events
                    a tuple of events means either event may have happened
        """
        # TODO: check that events are given in order!
        self.events = events
        self.num_targets = 10 #len(target_evts)

    def get_target_status(self):
        """
        @return ????
        """
        raise NotImplementedError()
        target_status = [None for i in range(self.num_targets)]
        for evt in self._uniq_events:
            target_status[evt.min_target + 1: evt.max_target] = self.UNRESOLVED
            target_status[evt.min_target] = self.INACTIVE
            target_status[evt.max_target] = self.INACTIVE
        for evt in self.unresolved_pos:
            target_status[evt.target] = self.UNRESOLVED
        return target_status

    def __str__(self):
        if self.events:
            return "...".join([str(evts) for evts in self.events])
        else:
            return "not_modified"


class BarcodeEventsRaw(BarcodeEvents):
    """
    In aaron's data, there are cases where there are multiple events associated with a single
    target. That doesn't make sense since each cut site can only be disturbed once.
    We will refer to these barcode event encodings as the `raw` version.
    """
    def __init__(self, events: List[Event]):
        self._target_evts = target_evts
        self._uniq_events = events
        self.organ = organ
