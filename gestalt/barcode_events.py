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
        targets: List[int] = []):
        """
        @param start_pos: position where event begins
        @param del_len: number of nucleotides deleted
        @param insert_str: sequence of nucleotides inserted
        @param target: which target this event
        """
        self.start_pos = start_pos
        self.del_len = del_len
        self.del_end = start_pos + del_len - 1
        self.insert_str = insert_str
        self.targets = targets

    def __str__(self):
        return self.get_str_id()

    def get_str_id(self):
        """
        Identifying string for this event
        """
        return "%d-%d, %s" % (self.start_pos, self.del_end, self.insert_str)


class BarcodeEvents:
    """
    Represents a barcode in event-encoding.
    Efficient for estimation procedures.

    Use this representation for cleaned barcode representation where each target
    can be associated with at most a single event.
    """
    def __init__(self, target_evts: List[int], events: List[Event], organ: CellTypeTree):
        """
        @param target_evts: for each target, the event idx associated
        @param events: list defining the event for each event idx
        @param organ: organ the barcode was sequenced from
        """
        self.target_evts = target_evts
        self.uniq_events = events
        self.organ = organ

    def get_str_id(self):
        return ".".join([evt.get_str_id() for evt in self.uniq_events])

    def can_be_parent(self, barcode_evts):
        """
        @param barcode: BarcodeEvents
                (I can't put it in the argument typing cause python3 typing is lame)
        @return whether this barcode can be a parent of this other barcode
        """
        raise NotImplementedError()

class BarcodeEventsRaw(BarcodeEvents):
    """
    In aaron's data, there are cases where there are multiple events associated with a single
    target. That doesn't make sense since each cut site can only be disturbed once.
    We will refer to these barcode event encodings as the `raw` version.
    """
    def __init__(self, target_evts: List[List[int]], events: List[Event], organ: CellTypeTree):
        self.target_evts = target_evts
        self.uniq_events = events
        self.organ = organ

