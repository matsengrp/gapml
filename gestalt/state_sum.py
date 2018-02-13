from typing import List, Tuple, Set

from indel_sets import TargetTract, AncState, TargetTractRepr
from barcode_metadata import BarcodeMetadata
from allele_events import AlleleEvents

class StateSum:
    def __init__(self, tts_list: List[TargetTractRepr]):
        self.tts_list = list(tts_list)

    def __str__(self):
        return " OR ".join([str(tts) for tts in self.tts_list])

    @staticmethod
    def create_for_observed_allele(allele: AlleleEvents, bcode_meta: BarcodeMetadata):
        """
        Create AncState for a leaf node
        """
        tts = ()
        for evt in allele.events:
            min_deact_target, max_deact_target = bcode_meta.get_min_max_deact_targets(evt)
            tts += (TargetTract(
                    min_deact_target,
                    evt.min_target,
                    evt.max_target,
                    max_deact_target), )
        return StateSum([TargetTractRepr(*tts)])
