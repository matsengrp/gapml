from typing import List, Tuple, Set

from indel_sets import TargetTract, AncState, TractRepr
from barcode_metadata import BarcodeMetadata
from allele_events import AlleleEvents

class StateSum:
    def __init__(self, tract_repr_list: List[TractRepr]):
        self.tract_repr_list = list(tract_repr_list)

    def __str__(self):
        return " OR ".join([str(tract_repr) for tract_repr in self.tract_repr_list])

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
        return StateSum([TractRepr(*tts)])
