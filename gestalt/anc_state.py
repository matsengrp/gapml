import itertools
import numpy as np
from typing import List
from functools import reduce

from allele_events import AlleleEvents
from indel_sets import IndelSet, SingletonWC
from target_status import TargetStatus, TargetDeactTract
from barcode_metadata import BarcodeMetadata

class AncState:
    """
    See AncState defined in the manuscript
    """
    def __init__(self, indel_set_list: List[IndelSet] = []):
        """
        @param indel_set_list: Each indel set must be a Wildcard or SingletonWC
        """
        self.indel_set_list = indel_set_list

    def __str__(self):
        if self.indel_set_list:
            return "..".join([str(d) for d in self.indel_set_list])
        else:
            return "unmod"

    def to_max_target_status(self):
        if len(self.indel_set_list) == 0:
            return TargetStatus()
        else:
            max_target = self.indel_set_list[-1].max_deact_target
            binary_status = np.zeros(max_target + 1)
            for indel_set in self.indel_set_list:
                binary_status[indel_set.min_deact_target:indel_set.max_deact_target + 1] = 1
            targ_stat = TargetStatus._binary_status_to_target_status(binary_status.tolist())
            return targ_stat

    @staticmethod
    def create_for_observed_allele(allele: AlleleEvents, bcode_meta: BarcodeMetadata):
        """
        Create AncState for a leaf node
        """
        indel_set_list = []
        for evt in allele.events:
            min_deact_target, max_deact_target = evt.get_min_max_deact_targets(bcode_meta)
            indel_set_list.append(
                SingletonWC(
                    evt.start_pos,
                    evt.del_len,
                    min_deact_target,
                    evt.min_target,
                    evt.max_target,
                    max_deact_target,
                    evt.insert_str))
        return AncState(indel_set_list)

    @staticmethod
    def intersect(anc_state1, anc_state2):
        if len(anc_state1.indel_set_list) == 0:
            return AncState()

        idx1 = 0
        idx2 = 0
        n1 = len(anc_state1.indel_set_list)
        n2 = len(anc_state2.indel_set_list)
        intersect_list = []
        while idx1 < n1 and idx2 < n2:
            indel_set1 = anc_state1.indel_set_list[idx1]
            indel_set2 = anc_state2.indel_set_list[idx2]

            if indel_set2.max_target < indel_set1.min_target:
                idx2 += 1
                continue
            elif indel_set1.max_target < indel_set2.min_target:
                idx1 += 1
                continue

            # Now we have overlapping events
            indel_sets_intersect = IndelSet.intersect(indel_set1, indel_set2)
            if indel_sets_intersect:
                intersect_list.append(indel_sets_intersect)

            # Increment counter
            if indel_set1.max_target < indel_set2.max_target:
                idx1 += 1
            else:
                idx2 += 1

        return AncState(intersect_list)

    def get_singleton_wcs(self):
        return [indel_set for indel_set in self.indel_set_list if indel_set.__class__ == SingletonWC]

    def get_singletons(self):
        return [sgwc.get_singleton() for sgwc in self.get_singleton_wcs()]

    def generate_possible_target_statuses(self):
        def _get_target_sub_statuses(indel_set: IndelSet):
            if indel_set.__class__ == SingletonWC:
                inner_wc = indel_set.inner_wc
                if inner_wc is not None:
                    deact_tract = TargetDeactTract(inner_wc.min_target, inner_wc.max_target)
                    sub_statuses = deact_tract.get_contained_target_statuses()
                else:
                    sub_statuses = [TargetStatus()]
                singleton_targ_stat = TargetStatus(TargetDeactTract(
                    indel_set.min_deact_target, indel_set.max_deact_target))
                return [singleton_targ_stat] + sub_statuses
            else:
                deact_tract = TargetDeactTract(indel_set.min_target, indel_set.max_target)
                return deact_tract.get_contained_target_statuses()

        if len(self.indel_set_list) == 0:
            return [TargetStatus()]

        partitioned_target_sub_statuses = [
                _get_target_sub_statuses(indel_set) for indel_set in self.indel_set_list]

        full_target_statuses = list(itertools.product(*partitioned_target_sub_statuses))

        if len(full_target_statuses) == 0:
            return [TargetStatus()]

        merged_target_statuses = [
            reduce(lambda x,y: x.merge(y), targ_stat_raw)
            for targ_stat_raw in full_target_statuses]

        return merged_target_statuses
