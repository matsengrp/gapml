import itertools
import numpy as np
from typing import List

from barcode_metadata import BarcodeMetadata
from indel_sets import TargetTract

class TargetDeactTract(tuple):
    def __new__(cls, min_deact_target, max_deact_target):
        return tuple.__new__(cls, (min_deact_target, max_deact_target))

    def __getnewargs__(self):
        return self

    @property
    def min_deact_target(self):
        return self[0]

    @property
    def max_deact_target(self):
        return self[1]

    def __str__(self):
        return "TargetDeact[%d,%d]" % (self.min_deact_target, self.max_deact_target)

    def get_contained_target_statuses(self):
        target_status_elems = [
            [0,1] for _ in range(self.max_deact_target - self.min_deact_target + 1)]
        target_statuses_prod = itertools.product(*target_status_elems)
        target_statuses = []
        for target_binary_repr in target_statuses_prod:
            deact_targs = []
            curr_deact_start_targ = None
            for idx, val in enumerate(target_binary_repr):
                if val == 1:
                    if curr_deact_start_targ is None:
                        curr_deact_start_targ = idx
                else:
                    if curr_deact_start_targ is not None:
                        deact_targs.append(TargetDeactTract(
                            self.min_deact_target + curr_deact_start_targ,
                            self.min_deact_target + idx - 1))
                        curr_deact_start_targ = None
            if curr_deact_start_targ is not None:
                deact_targs.append(TargetDeactTract(
                    self.min_deact_target + curr_deact_start_targ,
                    self.max_deact_target))
            target_statuses.append(
                    TargetStatus(*deact_targs))

        # Sort the target statuses by number of deactivated targets...
        # Maybe this will help computation since the matrix becomes upper triangular
        target_statuses = list(sorted(target_statuses, key = lambda x: x.num_deact_targets))

        return target_statuses

class TargetStatus(tuple):
    def __new__(cls, *target_deact_tracts):
        """
        @param target_deact_tracts: Tuple[TargetDeactTract]
        """
        for i in range(len(target_deact_tracts) - 1):
            assert target_deact_tracts[i].max_deact_target != target_deact_tracts[i + 1].min_deact_target - 1

        return tuple.__new__(cls, target_deact_tracts)

    def __getnewargs__(self):
        return self

    def __str__(self):
        return "TargetStatus%s" % super(TargetStatus, self).__str__()

    @property
    def deact_targets(self):
        deact_targs = []
        for deact_tract in self:
            deact_targs += list(range(deact_tract.min_deact_target, deact_tract.max_deact_target + 1))
        return deact_targs

    @property
    def num_deact_targets(self):
        return len(self.deact_targets)

    def merge(self, other_targ_stat):
        """
        This performs the most basic merge!
        """
        if len(self) == 0:
            return other_targ_stat
        if len(other_targ_stat) == 0:
            return self

        max_targets = max(other_targ_stat[-1].max_deact_target + 1, self[-1].max_deact_target + 1)
        binary_status = self._get_binary_status(max_targets)
        for deact_tract in other_targ_stat:
            #assert binary_status[deact_tract.min_deact_target] == 0
            #assert binary_status[deact_tract.max_deact_target] == 0
            binary_status[deact_tract.min_deact_target: deact_tract.max_deact_target + 1] = 1

        return TargetStatus._binary_status_to_target_status(binary_status.tolist())

    def add_target_tract(self, target_tract: TargetTract):
        """
        This ignores any ordering of the target tract (so will add the target tract even if
        the ordering that they are introduced makes no sense)
        @return TargetStatus that results after adding this `target_tract`
        """
        if len(self) == 0:
            deact_tract = TargetDeactTract(
                    target_tract.min_deact_target,
                    target_tract.max_deact_target)
            return TargetStatus(deact_tract)

        max_targets = max(target_tract.max_deact_target + 1, self[-1].max_deact_target + 1)
        binary_status = self._get_binary_status(max_targets)
        #assert binary_status[target_tract.min_target] == 0
        #assert binary_status[target_tract.max_target] == 0
        binary_status[target_tract.min_deact_target: target_tract.max_deact_target + 1] = 1
        return TargetStatus._binary_status_to_target_status(binary_status.tolist())

    def minus(self, orig_targ_stat):
        """
        @param orig_targ_stat: TargetStatus, the "original" target status
        @return the list of targets that were deactivated by this target status,
                where the `orig_targ_stat` was the original target status.
        """
        orig_targs = set(orig_targ_stat.deact_targets)
        self_targs = set(self.deact_targets)
        if self_targs >= orig_targs:
            return self_targs - orig_targs
        else:
            return set()

    def _get_binary_status(self, n_targets: int):
        """
        @return numpy array with 1 where the target is no longer active
        """
        binary_status = np.zeros(n_targets, dtype=int)
        for deact_tract in self:
            binary_status[deact_tract.min_deact_target: deact_tract.max_deact_target + 1] = 1
        return binary_status

    def get_active_targets(self, bcode_meta: BarcodeMetadata):
        """
        @return List[int] of active targets
        """
        inactive_status = self._get_binary_status(bcode_meta.n_targets)
        return np.where(inactive_status == 0)[0].tolist()

    def get_possible_target_tracts(self, bcode_meta: BarcodeMetadata):
        """
        @return List[TargetTract]
        """
        # Take one step from this TT group using two step procedure
        # 1. enumerate all possible start positions for target tract
        # 2. enumerate all possible end positions for target tract

        active_any_targs = self.get_active_targets(bcode_meta)
        n_any_targs = len(active_any_targs)
        # List possible starts of the target tracts
        all_starts = [[] for _ in range(n_any_targs)]
        for i0_prime, t0_prime in enumerate(active_any_targs):
            # Short left trim
            all_starts[i0_prime].append((t0_prime, t0_prime))
            if t0_prime > 0:
                # Long left trim
                all_starts[i0_prime].append((t0_prime - 1, t0_prime))

        # Create possible ends of the target tracts
        all_ends = [[] for i in range(n_any_targs)]
        for i1_prime, t1_prime in enumerate(active_any_targs):
            # Short right trim
            all_ends[i1_prime].append((t1_prime, t1_prime))
            if t1_prime < bcode_meta.n_targets - 1:
                # Allow a long right trim
                all_ends[i1_prime].append((t1_prime, t1_prime + 1))

        # Finally create all possible target tracts by combining possible start and ends
        tt_evts = set()
        for j, tt_starts in enumerate(all_starts):
            for k in range(j, n_any_targs):
                tt_ends = all_ends[k]
                for tt_start in tt_starts:
                    for tt_end in tt_ends:
                        tt_evt = TargetTract(tt_start[0], tt_start[1], tt_end[0], tt_end[1])
                        tt_evts.add(tt_evt)

        return list(tt_evts)

    @staticmethod
    def _binary_status_to_target_status(binary_status: List[int]):
        """
        Convert a binary status to a TargetStatus
        @param binary_status: 1 means inactive, 0 means active in this numpy array
        @return TargetStatus
        """
        deact_targs = []
        curr_deact_start_targ = None
        for idx, val in enumerate(binary_status):
            if val == 1:
                if curr_deact_start_targ is None:
                    curr_deact_start_targ = idx
            else:
                if curr_deact_start_targ is not None:
                    deact_targs.append(TargetDeactTract(
                        curr_deact_start_targ,
                        idx - 1))
                    curr_deact_start_targ = None
        if curr_deact_start_targ is not None:
            deact_targs.append(TargetDeactTract(
                curr_deact_start_targ,
                len(binary_status) - 1))
        return TargetStatus(*deact_targs)

    @staticmethod
    def get_all_transitions(bcode_meta: BarcodeMetadata):
        """
        @return tuple of two Dicts:
            1. Dict[start TargetStatus, Dict[end TargetStatus, List[TargetTract] that can be introduced to the start TargetStatus to create the end TargetStatus]
            2. Dict[end TargetStatus, Set[start TargetStatus]]: maps each end target status to all the possible start target statuses (within one step)
        """
        target_status_transition_dict = dict()
        target_status_inverse_transition_dict = dict()
        deact_targs = TargetDeactTract(0, bcode_meta.n_targets - 1)
        target_statuses = deact_targs.get_contained_target_statuses()
        for targ_stat in target_statuses:
            targ_stat_start_dict = dict()
            active_targets = targ_stat.get_active_targets(bcode_meta)
            possible_target_tracts = targ_stat.get_possible_target_tracts(bcode_meta)
            for target_tract in possible_target_tracts:
                new_targ_stat = targ_stat.add_target_tract(target_tract)
                if new_targ_stat in targ_stat_start_dict:
                    targ_stat_start_dict[new_targ_stat].append(target_tract)
                else:
                    targ_stat_start_dict[new_targ_stat] = [target_tract]

                # Also update the inverse dictionary -- maps new state to possible previous states
                if new_targ_stat in target_status_inverse_transition_dict:
                    target_status_inverse_transition_dict[new_targ_stat].add(targ_stat)
                else:
                    target_status_inverse_transition_dict[new_targ_stat] = set([targ_stat])

            target_status_transition_dict[targ_stat] = targ_stat_start_dict

        return target_status_transition_dict, target_status_inverse_transition_dict
