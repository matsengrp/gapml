from enum import Enum
from typing import List
from typing import Dict

class EventType(Enum):
    DELETE = 1
    INSERT = 2
    UNKNOWN = 3


class Event:
    def __init__(self, event_type: EventType, event_len: int, start_pos: int):
        self.event_type = event_type
        self.event_len = event_len
        self.start_pos = start_pos

    def __str__(self):
        return self.get_evt_id()

    def get_evt_id(self):
        """
        Identifying string for this event
        """
        raise NotImplementedError()


class DeletionEvent(Event):
    def __init__(self, event_len: int, start_pos: int):
        super(DeletionEvent, self).__init__(EventType.DELETE, event_len, start_pos)

    def get_evt_id(self):
        return "%s: %d+%d" % (self.event_type, self.start_pos, self.event_len)


class InsertionEvent(Event):
    def __init__(self, event_len: int, start_pos: int, insert_str: str):
        super(InsertionEvent, self).__init__(EventType.INSERT, event_len, start_pos)
        self.insert_str = insert_str

    def get_evt_id(self):
        return "%s: %d+%d, %s" % (self.event_type, self.start_pos, self.event_len, self.insert_str)


class BarcodeEvents:
    def __init__(self, events: List[Event], organ: str):
        self.events = events
        self.organ = organ


class CellReads:
    def __init__(self, all_barcodes: List[BarcodeEvents]):
        """
        @param all_barcodes: all the barcodes in the cell reads data
        """
        self.all_barcodes = all_barcodes

    def get_event_abundance(self):
        """
        @return dictionary mapping event id to number of times the event was seen
        """
        evt_weight_dict = dict()
        for barcode_evts in self.all_barcodes:
            for evt in barcode_evts.events:
                evt_id = evt.get_evt_id()
                if evt_id not in evt_weight_dict:
                    evt_weight_dict[evt_id] = 1
                else:
                    evt_weight_dict[evt_id] += 1
        return evt_weight_dict
