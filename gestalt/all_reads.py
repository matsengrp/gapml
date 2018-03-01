from typing import List, Dict

from cell_state import CellTypeTree
from allele_events import AlleleEvents

class CellRead:
    def  __init__(self, allele: AlleleEvents, organ: CellTypeTree):
        # TODO: make organ an int?
        self.allele = allele
        self.organ = organ

    @property
    def events(self):
        return self.allele.events

class CellReads:
    def __init__(self, reads: List[CellRead], organ_dict: Dict[str, str]):
        """
        @param organ_dict: provides a way to map between an organ id and organ name
        """
        self.reads = reads
        self.organ_dict = organ_dict
        self.event_abundance = self._get_event_abundance()
        self.uniq_events = set([evt for allele_evts in self.reads for evt in allele_evts.events])
        self.uniq_alleles = self._get_uniq_alleles()

    def _get_uniq_alleles(self):
        uniq_alleles = []
        uniq_strs = set()
        for allele in self.reads:
            allele_id = str(allele)
            if allele_id in uniq_strs:
                continue
            else:
                uniq_strs.add(allele_id)
                uniq_alleles.append(allele)
        return uniq_alleles

    def _get_event_abundance(self):
        """
        @return dictionary mapping event id to number of times the event was seen
        """
        evt_weight_dict = dict()
        for allele_evts in self.reads:
            for evt in allele_evts.events:
                evt_id = evt
                if evt_id not in evt_weight_dict:
                    evt_weight_dict[evt_id] = 1
                else:
                    evt_weight_dict[evt_id] += 1
        return evt_weight_dict
