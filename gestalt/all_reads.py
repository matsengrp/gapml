from typing import List, Dict

from cell_state import CellTypeTree
from barcode_events import BarcodeEvents

class CellRead:
    def  __init__(self, barcode: BarcodeEvents, organ: CellTypeTree):
        # TODO: make organ an int?
        self.barcode = barcode
        self.organ = organ

    @property
    def events(self):
        return self.barcode.events

class CellReads:
    def __init__(self, all_barcodes: List[BarcodeEvents], organ_dict: Dict[CellTypeTree, str]):
        """
        @param all_barcodes: all the barcodes in the cell reads data
        """
        self.all_barcodes = all_barcodes
        self.organ_dict = organ_dict
        self.event_abundance = self._get_event_abundance()
        self.uniq_events = set([evt for barcode_evts in self.all_barcodes for evt in barcode_evts.events])
        self.uniq_barcodes = self._get_uniq_barcodes()

    def _get_uniq_barcodes(self):
        uniq_barcodes = []
        uniq_strs = set()
        for bcode in self.all_barcodes:
            barcode_id = str(bcode)
            if barcode_id in uniq_strs:
                continue
            else:
                uniq_strs.add(barcode_id)
                uniq_barcodes.append(bcode)
        return uniq_barcodes

    def _get_event_abundance(self):
        """
        @return dictionary mapping event id to number of times the event was seen
        """
        evt_weight_dict = dict()
        for barcode_evts in self.all_barcodes:
            for evt in barcode_evts.events:
                evt_id = str(evt)
                if evt_id not in evt_weight_dict:
                    evt_weight_dict[evt_id] = 1
                else:
                    evt_weight_dict[evt_id] += 1
        return evt_weight_dict
