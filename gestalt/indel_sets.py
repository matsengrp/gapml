from typing import List, Dict, Tuple
from functools import reduce

from allele_events import AlleleEvents
from barcode_metadata import BarcodeMetadata

class IndelSet(tuple):
    """
    A superclass for wildcard and singleton-wildcards
    """
    def __new__(cls):
        # Don't call this by itself
        raise NotImplementedError()

    @staticmethod
    def intersect(indel_set1, indel_set2):
        if indel_set1 == indel_set2:
            return indel_set1
        else:
            wc1 = indel_set1.inner_wc if indel_set1.__class__ == SingletonWC else indel_set1
            wc2 = indel_set2.inner_wc if indel_set2.__class__ == SingletonWC else indel_set2
            if wc1 is None or wc2 is None:
                return None
            else:
                return Wildcard.intersect(wc1, wc2)

class Wildcard(IndelSet):
    def __new__(cls,
            min_target: int,
            max_target: int):
        return tuple.__new__(cls, (min_target, max_target))

    def __getnewargs__(self):
        return (self.min_target, self.max_target)

    @property
    def min_target(self):
        return self[0]

    @property
    def min_deact_target(self):
        return self.min_target

    @property
    def max_target(self):
        return self[1]

    @property
    def max_deact_target(self):
        return self.max_target

    @property
    def inner_wc(self):
        return self

    def get_singleton(self):
        return None

    @staticmethod
    def intersect(wc1, wc2):
        min_targ = max(wc1.min_target, wc2.min_target)
        max_targ = min(wc1.max_target, wc2.max_target)
        return Wildcard(min_targ, max_targ)

class SingletonWC(IndelSet):
    """
    Actually the same definition as an Event right now....
    TODO: do not repeat code?
    """
    def __new__(cls,
            start_pos: int,
            del_len: int,
            min_deact_target: int,
            min_target: int,
            max_target: int,
            max_deact_target: int,
            insert_str: str = ""):
        return tuple.__new__(cls, (start_pos, del_len, min_deact_target, min_target, max_target, max_deact_target, insert_str))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.min_deact_target, self.min_target, self.max_target, self.max_deact_target, self.insert_str)

    @property
    def start_pos(self):
        return self[0]

    @property
    def del_len(self):
        return self[1]

    @property
    def del_end(self):
        return self.start_pos + self.del_len

    @property
    def min_deact_target(self):
        return self[2]

    @property
    def min_target(self):
        return self[3]

    @property
    def max_target(self):
        return self[4]

    @property
    def max_deact_target(self):
        return self[5]

    @property
    def insert_str(self):
        return self[6]

    @property
    def insert_len(self):
        return len(self.insert_str)

    # TODO: make this a function instead?
    @property
    def inner_wc(self):
        if self.max_target - 1 >= self.min_target + 1:
            return Wildcard(self.min_target + 1, self.max_target - 1)
        else:
            return None

    def get_singleton(self):
        return Singleton(self.start_pos, self.del_len, self.min_deact_target, self.min_target, self.max_target, self.max_deact_target, self.insert_str)

class Singleton(IndelSet):
    """
    Actually the same definition as an Event right now....
    Actually a singleton now... so same as event?
    TODO: do not repeat code?
    """
    def __new__(cls,
            start_pos: int,
            del_len: int,
            min_deact_target: int,
            min_target: int,
            max_target: int,
            max_deact_target: int,
            insert_str: str = ""):
        return tuple.__new__(cls, (start_pos, del_len, min_deact_target, min_target, max_target, max_deact_target, insert_str))

    def __getnewargs__(self):
        return (self.start_pos, self.del_len, self.min_deact_target, self.min_target, self.max_target, self.max_deact_target, self.insert_str)

    @property
    def start_pos(self):
        return self[0]

    @property
    def del_len(self):
        return self[1]

    @property
    def del_end(self):
        return self.start_pos + self.del_len

    @property
    def min_deact_target(self):
        return self[2]

    @property
    def min_target(self):
        return self[3]

    @property
    def max_target(self):
        return self[4]

    @property
    def max_deact_target(self):
        return self[5]

    @property
    def insert_str(self):
        return self[6]

    @property
    def insert_len(self):
        return len(self.insert_str)

    @property
    def is_left_long(self):
        return self.min_deact_target != self.min_target

    @property
    def is_right_long(self):
        return self.max_deact_target != self.max_target

    def get_target_tract(self):
        return TargetTract(
                self.min_deact_target,
                self.min_target,
                self.max_target,
                self.max_deact_target)

class DeactEvt(IndelSet):
    def __new__(cls):
        # Don't call this by itself
        raise NotImplementedError()

    def get_deact_result(self):
        raise NotImplementedError()

class Tract(IndelSet):
    def __new__(cls):
        # Don't call this by itself
        raise NotImplementedError()

    @property
    def min_deact_target(self):
        return self[0]

    @property
    def max_deact_target(self):
        return self[-1]

class TargetTract(Tract, DeactEvt):
    def __new__(cls,
            min_deact_targ: int,
            min_targ: int,
            max_targ: int,
            max_deact_targ: int):
        return tuple.__new__(cls, (min_deact_targ, min_targ, max_targ, max_deact_targ))

    def __getnewargs__(self):
        return (self.min_deact_targ, self.min_targ, self.max_targ, self.max_deact_targ)

    @property
    def min_target(self):
        return self[1]

    @property
    def max_target(self):
        return self[2]

    @property
    def is_left_long(self):
        return self.min_deact_target != self.min_target

    @property
    def is_right_long(self):
        return self.max_deact_target != self.max_target
    
    @property
    def is_target_tract(self):
        return True

    @property
    def is_deact_tract(self):
        return False

    def get_deact_result(self):
        return self

class DeactTract(Tract):
    def __new__(cls, min_deact_target, max_deact_target):
        return tuple.__new__(cls, (min_deact_target, max_deact_target))

    def __getnewargs__(self):
        return self

    @property
    def is_deact_tract(self):
        return True

class DeactTargetsEvt(DeactEvt):
    def __new__(cls, *deact_targs):
        return tuple.__new__(cls, deact_targs)

    def __getnewargs__(self):
        return self

    @property
    def min_deact_target(self):
        return self[0]

    @property
    def max_deact_target(self):
        return self[-1]

    @property
    def is_target_tract(self):
        return False

    def get_deact_result(self):
        return DeactTract(self.min_deact_target, self.max_deact_target)

class AncState:
    def __init__(self, indel_set_list: List[IndelSet] = []):
        self.indel_set_list = indel_set_list

    def __str__(self):
        if self.indel_set_list:
            return "..".join([str(d) for d in self.indel_set_list])
        else:
            return "unmod"

    @staticmethod
    def create_for_observed_allele(allele: AlleleEvents, bcode_meta: BarcodeMetadata):
        """
        Create AncState for a leaf node
        """
        indel_set_list = []
        for evt in allele.events:
            min_deact_target, max_deact_target = bcode_meta.get_min_max_deact_targets(evt)
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
            if indel_set1.max_target > indel_set2.max_target:
                idx1 += 1
            else:
                idx2 += 1

        return AncState(intersect_list)

class TractRepr(tuple):
    def __new__(cls, *args):
        return tuple.__new__(cls, args)

    def __getnewargs__(self):
        return self

    def lesseq(self, tracts2):
        """
        @param tracts2: TargetTractRepr

        @return self <= tracts2
        """
        idx2 = 0
        n2 = len(tracts2)
        for tract1 in self:
            while idx2 < n2:
                tract2 = tracts2[idx2]
                if tract1 == tract2:
                    idx2 += 1
                    break
                elif tract1.min_deact_target > tract2.min_deact_target and tract1.max_deact_target < tract2.max_deact_target:
                    # If tract1 < tract2
                    idx2 += 1
                    break
                elif tract2.max_deact_target < tract1.min_deact_target:
                    idx2 += 1
                else:
                    return False
        return True

    def diff(self, tracts2):
        """
        @param tracts2: TargetTractRepr

        @return tracts2 - self if self <= tracts2, otherwise returns empty tuple
        """
        if not self.lesseq(tracts2):
            return ()

        idx1 = 0
        idx2 = 0
        n1 = len(self)
        n2 = len(tracts2)
        new_tuple = ()
        while idx1 < n1 and idx2 < n2:
            tract1 = self[idx1]
            tract2 = tracts2[idx2]

            if tract2.max_deact_target < tract1.min_deact_target:
                new_tuple += (tract2,)
                idx2 += 1
                continue

            # Now we have overlapping events
            idx1 += 1
            idx2 += 1
            if tract1 != tract2:
                new_tuple += (tract2,)

        new_tuple += tracts2[idx2:]

        return new_tuple

    @staticmethod
    def merge(tract_groups: List[Tuple[TargetTract]]):
        """
        @return flatractened version of a list of tuples of target tract
        """
        tracts_raw = reduce(lambda x,y: x + y, tract_groups, ())
        return TractRepr(*tracts_raw)

def merge_tracts(tract_group_orig: Tuple[Tract], tract_result: Tract):
    """
    @param tract_group_orig: a tuple of tracts
    @param tract_result: the tract that is getting introduced

    @return the new tract tuple after introducing this tract (deals with masking
            and merging of adjacent deactivation tracts)
    """
    # Create the tuple of tracts that occurs after deact_evt is introduced
    # Calls python sort which is inefficient maybe, but it's cleaner code
    # TODO: make faster?
    # TODO: write tests for this function
    tract_tuple = tuple(sorted(
        tract_group_orig + (tract_result,),
        key=lambda tract: tract.min_deact_target))

    # Merge adjacent deactivation tracts
    tract_tuple_merged = ()
    curr_deact_tract = None
    prev_tract = None
    for tract in tract_tuple:
        if prev_tract is not None:
            # Check if there are masked tracts
            if prev_tract.max_deact_target > tract.max_deact_target:
                # Check that the masked tract is not the new tract
                assert(tract != tract_result)
                continue

        if not tract.is_deact_tract:
            if curr_deact_tract:
                tract_tuple_merged += (curr_deact_tract, )
                curr_deact_tract = None

            tract_tuple_merged += (tract,)
        else:
            if curr_deact_tract is None:
                curr_deact_tract = tract
            elif tract.min_deact_target == curr_deact_tract.max_deact_target + 1:
                curr_deact_tract = DeactTract(
                        curr_deact_tract.min_deact_target,
                        tract.max_deact_target)
            else:
                tract_tuple_merged += (curr_deact_tract, )
                curr_deact_tract = tract
        prev_tract = tract

    # If there is a lingering deact tract to add to the list...
    if curr_deact_tract:
        tract_tuple_merged += (curr_deact_tract, )
    
    return tract_tuple_merged

def get_deactivated_targets(tract_grp: Tuple[IndelSet]):
    if tract_grp:
        deactivated = list(range(tract_grp[0].min_deact_target, tract_grp[0].max_deact_target + 1))
        for tract in tract_grp[1:]:
            deactivated += list(range(tract.min_deact_target, tract.max_deact_target + 1))
        return set(deactivated)
    else:
        return set()