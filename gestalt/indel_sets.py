from typing import List, Dict, Tuple
from functools import reduce

from allele_events import AlleleEvents
from barcode_metadata import BarcodeMetadata

class IndelSet(tuple):
    """
    A superclass for things that represent sets of Indels
    """
    def __new__(cls):
        # Don't call this by itself
        raise NotImplementedError()

    @staticmethod
    def intersect(indel_set1, indel_set2):
        if indel_set1 == indel_set2:
            return indel_set1
        else:
            wc1 = indel_set1.inner_wc
            wc2 = indel_set2.inner_wc
            if wc1 is not None and wc1.min_target <= indel_set2.min_deact_target and indel_set2.max_deact_target <= wc1.max_target:
                return indel_set2
            elif wc2 is not None and wc2.min_target <= indel_set1.min_deact_target and indel_set1.max_deact_target <= wc2.max_target:
                return indel_set1
            else:
                return Wildcard.intersect(wc1, wc2)

class Wildcard(IndelSet):
    """
    See definition of Wildcard in the manuscript
    """
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
        if wc1 is None or wc2 is None:
            return None
        min_targ = max(wc1.min_target, wc2.min_target)
        max_targ = min(wc1.max_target, wc2.max_target)
        return Wildcard(min_targ, max_targ)

class SingletonWC(IndelSet):
    """
    See definition of Singleton-wildcard in manuscript
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
    def inner_wc(self):
        if self.max_target - 1 >= self.min_target + 1:
            return Wildcard(self.min_target + 1, self.max_target - 1)
        else:
            return None

    def get_singleton(self):
        return Singleton(self.start_pos, self.del_len, self.min_deact_target, self.min_target, self.max_target, self.max_deact_target, self.insert_str)

class Singleton(IndelSet):
    """
    This represents an indel set containing a single indel event

    Actually the same definition as an Event right now....
    Think about not duplicating code?
    """
    def __new__(cls,
            start_pos: int,
            del_len: int,
            min_deact_target: int,
            min_target: int,
            max_target: int,
            max_deact_target: int,
            insert_str: str = ""):
        assert min_deact_target >= 0
        assert min_deact_target >= min_target - 1
        assert min_deact_target <= min_target
        assert min_target <= max_target
        assert max_target <= max_deact_target
        assert max_target + 1 >= max_deact_target
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
    def is_long_del(self):
        return [self.is_left_long, self.is_right_long]

    @property
    def is_left_long(self):
        return self.min_deact_target != self.min_target

    @property
    def is_right_long(self):
        return self.max_deact_target != self.max_target

    @property
    def is_intertarget(self):
        return self.max_target != self.min_target

    def get_target_tract(self):
        return TargetTract(
                self.min_deact_target,
                self.min_target,
                self.max_target,
                self.max_deact_target)

    def get_trim_lens(self, bcode_meta: BarcodeMetadata):
        left_trim_len = bcode_meta.abs_cut_sites[self.min_target] - self.start_pos
        right_trim_len = self.del_end - bcode_meta.abs_cut_sites[self.max_target]
        assert left_trim_len >= 0
        assert right_trim_len >= 0
        assert left_trim_len <= bcode_meta.left_max_trim[self.min_target]
        assert right_trim_len <= bcode_meta.right_max_trim[self.max_target]
        return left_trim_len, right_trim_len

    def __str__(self):
        return "Singleton,%d,%d,%d,%d,%d,%d,%s" % (
                self.start_pos,
                self.del_len,
                self.min_deact_target,
                self.min_target,
                self.max_target,
                self.max_deact_target,
                self.insert_str)

class DeactEvt(IndelSet):
    """
    An event that will deactivate targets
    Implements a function that specifies what targets are deactivated
    """
    def __new__(cls):
        # Don't call this by itself
        raise NotImplementedError()

    def get_deact_result(self):
        raise NotImplementedError()

class Tract(IndelSet):
    """
    A tract is an object with a min and max deactivated target
    """
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
    """
    See definition in the manuscript
    """
    def __new__(cls,
            min_deact_targ: int,
            min_targ: int,
            max_targ: int,
            max_deact_targ: int):
        assert min_targ <= max_targ
        assert min_deact_targ >= min_targ - 1
        assert min_deact_targ <= min_targ
        assert max_targ <= max_deact_targ
        assert max_deact_targ <= max_targ + 1
        return tuple.__new__(cls, (min_deact_targ, min_targ, max_targ, max_deact_targ))

    def __getnewargs__(self):
        return self

    @property
    def min_deact_targ(self):
        return self[0]

    @property
    def min_target(self):
        return self[1]

    @property
    def max_target(self):
        return self[2]

    @property
    def max_deact_targ(self):
        return self[3]

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
    """
    Stores min and max deactivated targets
    Object for easy computation in the code
    """
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

class TractRepr(tuple):
    """
    Look up "target tract representation" in the manuscript
    Essentially a tuple of target tracts
    (Used as an allele group to sum over in the likelihood calculation)
    """
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
