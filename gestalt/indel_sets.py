from typing import List
from typing import Dict

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

class TargetTract(IndelSet):
    def __new__(cls,
            min_deact_targ: int,
            min_targ: int,
            max_targ: int,
            max_deact_targ: int):
        return tuple.__new__(cls, (min_deact_targ, min_targ, max_targ, max_deact_targ))

    def __getnewargs__(self):
        return (self.min_deact_targ, self.min_targ, self.max_targ, self.max_deact_targ)

    @property
    def min_deact_target(self):
        return self[0]

    @property
    def min_target(self):
        return self[1]

    @property
    def max_target(self):
        return self[2]

    @property
    def max_deact_target(self):
        return self[3]

    @property
    def is_left_long(self):
        return self.min_deact_target != self.min_target

    @property
    def is_right_long(self):
        return self.max_deact_target != self.max_target

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

class TargetTractRepr(tuple):
    def __new__(cls, *args):
        return tuple.__new__(cls, args)

    def __getnewargs__(self):
        return self

    def lesseq(self, tts2):
        """
        @param tts2: TargetTractRepr

        @return self <= tts2
        """
        idx2 = 0
        n2 = len(tts2)
        for tt1 in self:
            while idx2 < n2:
                tt2 = tts2[idx2]
                if tt1 == tt2:
                    idx2 += 1
                    break
                elif tt1.min_deact_target > tt2.min_target and tt1.max_deact_target < tt2.max_target:
                    # If tt1 < tt2
                    idx2 += 1
                    break
                elif tt2.max_deact_target < tt1.min_deact_target:
                    idx2 += 1
                else:
                    return False
        return True

    def diff(self, tts2):
        """
        @param tts2: TargetTractRepr

        @return tts2 - self if self <= tts2, otherwise returns empty tuple
        """
        if not self.lesseq(tts2):
            return ()

        idx1 = 0
        idx2 = 0
        n1 = len(self)
        n2 = len(tts2)
        new_tuple = ()
        while idx1 < n1 and idx2 < n2:
            tt1 = self[idx1]
            tt2 = tts2[idx2]

            if tt2.max_target < tt1.min_target:
                new_tuple += (tt2,)
                idx2 += 1
                continue

            # Now we have overlapping events
            idx1 += 1
            idx2 += 1
            if tt1 != tt2:
                new_tuple += (tt2,)

        new_tuple += tts2[idx2:]

        return new_tuple
