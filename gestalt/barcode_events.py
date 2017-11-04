from enum import Enum
from typing import List
from typing import Dict

from cell_state import CellTypeTree
from constants import NUM_BARCODE_V7_TARGETS

"""
Objects for representing a barcode using an event-encoded format
"""


class EventType(Enum):
    DELETE = 1
    INSERT = 2


class Event:
    def __init__(
        self,
        start_pos: int,
        del_len: int,
        insert_str: str,
        targets: List[int] = []):
        """
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
    Also matches the encoding in Aaron's files
    """
    def __init__(self, target_evts: List[List[int]], events: List[Event], organ: CellTypeTree):
        """
        @param target_evts: for each target, the list of event idxs associated
        TODO: Technically a single target cannot be associated with multiple events.
            This is here because Aaron's alignment file give multiple events per target
            We should probably do something about this...
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
