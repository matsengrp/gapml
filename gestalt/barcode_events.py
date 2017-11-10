from enum import Enum
from typing import List
from typing import Dict

from cell_state import CellTypeTree
from constants import NUM_BARCODE_V7_TARGETS

"""
Objects for representing a barcode using an event-encoded format
"""


class Event:
    def __init__(
        self,
        start_pos: int,
        del_len: int,
        insert_str: str,
        targets: List[int]):
        """
        @param start_pos: position where event begins
        @param del_len: number of nucleotides deleted
        @param insert_str: sequence of nucleotides inserted
        @param target: which target this event is associated with
        """
        self.start_pos = start_pos
        self.del_len = del_len
        self.del_end = start_pos + del_len - 1
        self.insert_str = insert_str
        self.targets = targets
        self.min_target = min(targets)
        self.max_target = max(targets)
        self.is_focal = self.min_target == self.max_target

    def is_equal(self, evt):
        return (self.start_pos == evt.start_pos
                and self.del_len == evt.del_len
                and self.insert_str == evt.insert_str)

    def get_str_id(self):
        """
        Identifying string for this event
        """
        return "(%d-%d, %s)" % (self.start_pos, self.del_end, self.insert_str)

    def __str__(self):
        return self.get_str_id()


class PlaceholderEvent(Event):
    def __init__(self, is_focal: bool, target: int):
        """
        just create a placeholder event
        """
        self.is_focal = is_focal
        self.targets = [target]

    def is_equal(self, evt):
        return False

    def get_str_id(self):
        """
        Identifying string for this event
        """
        return "??"

    def __str__(self):
        return self.get_str_id()


class BarcodeEvents:
    """
    Represents a barcode in event-encoding.
    Efficient for estimation procedures.

    Use this representation for cleaned barcode representation where each target
    can be associated with at most a single event.
    """
    def __init__(self, target_evts: List, events: List[Event], organ: CellTypeTree):
        """
        @param target_evts: for each target, the event idx associated,
                            idx of the event if an event occurred
                            None if no event occurred
        @param events: list defining the event for each event idx
        @param organ: organ the barcode was sequenced from
        """
<<<<<<< HEAD
        # These are private objects! Do not modify directly!
        self._target_evts = target_evts
        self._uniq_events = events
        self.organ = organ
        self.num_targets = len(target_evts)
        assert(self.num_targets == 10)

    def add_event(self):
        raise NotImplementedError()

    def get_uniq_events(self):
        return self._uniq_events

    def get_event(self, target_idx: int):
        """
        @return the event associated with this target idx
        """
        target_evt_idx = self._target_evts[target_idx]
        if target_evt_idx is not None:
            return self._uniq_events[target_evt_idx]
        else:
            return None

    def get_target_status(self):
        """
        @return a boolean array to indicate which targets are active (aka can be cut)
        """
        return [1 if self._target_evts[i] else 0 for i in range(self.num_targets)]

    def get_str_id(self):
        """
        Generates a string based on event details
        """
        return "...".join([evt.get_str_id() for evt in self._uniq_events])

    def __str__(self):
        return self.get_str_id()


class BarcodeEventsRaw(BarcodeEvents):
    """
    In aaron's data, there are cases where there are multiple events associated with a single
    target. That doesn't make sense since each cut site can only be disturbed once.
    We will refer to these barcode event encodings as the `raw` version.
    """
    def __init__(self, target_evts: List[List[int]], events: List[Event], organ: CellTypeTree):
        self._target_evts = target_evts
        self._uniq_events = events
        self.organ = organ
