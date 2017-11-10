from typing import List

from barcode_events import BarcodeEvents


class CellReads:
    def __init__(self, all_barcodes: List[BarcodeEvents]):
        """
        @param all_barcodes: all the barcodes in the cell reads data
        """
        self.all_barcodes = all_barcodes
        self.event_abundance = self._get_event_abundance()
        self.event_str_ids = self.event_abundance.keys()
        self.uniq_barcodes = self._get_uniq_barcodes()

    def _get_uniq_barcodes(self):
        uniq_barcodes = []
        uniq_strs = set()
        for bcode in self.all_barcodes:
            barcode_id = bcode.get_str_id()
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
            for evt in barcode_evts.get_uniq_events():
                evt_id = evt.get_str_id()
                if evt_id not in evt_weight_dict:
                    evt_weight_dict[evt_id] = 1
                else:
                    evt_weight_dict[evt_id] += 1
        return evt_weight_dict
