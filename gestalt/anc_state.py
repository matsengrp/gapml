import itertools
import numpy as np
from typing import List
from functools import reduce

from allele_events import AlleleEvents
from indel_sets import IndelSet, SingletonWC, Wildcard
from target_status import TargetStatus, TargetTractTuple, TargetDeactTract
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
        """
        @return the max target status associated with the anc state (including both wildcards and singleton-wildcards)
        """
        if len(self.indel_set_list) == 0:
            return TargetStatus()
        else:
            max_target = self.indel_set_list[-1].max_deact_target
            binary_status = np.zeros(max_target + 1)
            for indel_set in self.indel_set_list:
                binary_status[indel_set.min_deact_target:indel_set.max_deact_target + 1] = 1
            targ_stat = TargetStatus._binary_status_to_target_status(binary_status.tolist())
            return targ_stat

    def to_sg_max_target_status(self):
        """
        @return the max target status associated with ONLY the singleton-wildcards in the anc state
        """
        if len(self.indel_set_list) == 0:
            return TargetStatus()
        else:
            max_target = self.indel_set_list[-1].max_deact_target
            binary_status = np.zeros(max_target + 1)
            for indel_set in self.indel_set_list:
                if indel_set.__class__ == SingletonWC:
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

            if indel_set2.max_deact_target < indel_set1.min_deact_target:
                idx2 += 1
                continue
            elif indel_set1.max_deact_target < indel_set2.min_deact_target:
                idx1 += 1
                continue

            # Now we have overlapping events
            indel_sets_intersect = IndelSet.intersect(indel_set1, indel_set2)
            if indel_sets_intersect:
                intersect_list.append(indel_sets_intersect)

            # Increment counter
            if indel_set1.max_deact_target < indel_set2.max_deact_target:
                idx1 += 1
            else:
                idx2 += 1

        return AncState(intersect_list)

    def get_singleton_wcs(self):
        return [indel_set for indel_set in self.indel_set_list if indel_set.__class__ == SingletonWC]

    def get_singletons(self):
        return [sgwc.get_singleton() for sgwc in self.get_singleton_wcs()]

    def generate_possible_target_statuses(self):
        """
        @return List[TargetStatus] for all target statuses that are possible
                with that ancestral state
        """
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

    def is_possible(self, target_tract_tuple: TargetTractTuple):
        """
        @return boolean whether possible
        """
        singletons = self.get_singleton_wcs()
        sg_tts = {sg.get_target_tract(): sg for sg in singletons}

        fake_anc_state = []
        for target_tract in target_tract_tuple:
            if target_tract in sg_tts:
                fake_anc_state.append(sg_tts[target_tract])
            else:
                fake_anc_state.append(Wildcard(
                    target_tract.min_deact_targ,
                    target_tract.max_deact_targ))
        intersected_anc_state = AncState.intersect(self, AncState(fake_anc_state)).indel_set_list
        if len(intersected_anc_state) != len(fake_anc_state):
            return False
        else:
            for fake_indel_set, intersected_indel_set in zip(fake_anc_state, intersected_anc_state):
                if fake_indel_set != intersected_indel_set:
                    return False
        return True
