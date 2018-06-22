import itertools

from barcode_metadata import BarcodeMetadata

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
        return target_statuses

class TargetStatus(tuple):
    def __new__(cls, *target_deact_tracts):
        """
        @param target_deact_tracts: Tuple[TargetDeactTract]
        """
        return tuple.__new__(cls, target_deact_tracts)

    def __getnewargs__(self):
        return self

    def __str__(self):
        return "TargetStatus%s" % super(TargetStatus, self).__str__()

    @property
    def deact_targets(self):
        deact_targs = []
        for deact_tract in self:
            deact_targs += list(range(deact_tract.min_deact_target, deact_tract.max_deact_target))
        return deact_targs

    def merge(self, other_targ_stat):
        my_deact_tracts = list(self)
        other_deact_tracts = list(other_targ_stat)
        new_targ_stat = list(sorted(my_deact_tracts + other_deact_tracts, key = lambda x: x.min_deact_target))
        return TargetStatus(*new_targ_stat)

    def add(self, deact_tract: TargetDeactTract):
        my_deact_tracts = list(self)
        new_targ_stat = list(sorted(my_deact_tracts + [deact_tract], key = lambda x: x.min_deact_target))
        return TargetStatus(*new_targ_stat)

    def minus(self, other_targ_stat):
        return set(self.deact_targets) - set(other_targ_stat.deact_targets)

    def get_active_targets(self, bcode_meta: BarcodeMetadata):
        if len(self) == 0:
            return list(range(bcode_meta.n_targets))

        active_targets = list(range(self[0].min_deact_target))
        last_deact_targ = self[0].max_deact_target
        for deact_targs in self[1:]:
            active_targets += list(range(last_deact_targ + 1, deact_targs.min_deact_target))
            last_deact_targ = deact_targs.max_deact_target
        active_targets += list(range(last_deact_targ + 1, bcode_meta.n_targets))
        return active_targets 

    @staticmethod
    def get_all_transitions(bcode_meta: BarcodeMetadata):
        target_status_transition_dict = dict()
        deact_targs = TargetDeactTract(0, bcode_meta.n_targets - 1)
        target_statuses = deact_targs.get_contained_target_statuses()
        for targ_stat in target_statuses:
            target_status_transition_dict[targ_stat] = dict()
            active_targets = targ_stat.get_active_targets(bcode_meta)
            for i, start_targ in enumerate(active_targets):
                for end_targ in active_targets[i:]:
                    transition_deact_tract = TargetDeactTract(start_targ, end_targ)
                    new_targ_stat = targ_stat.add(transition_deact_tract)
                    target_status_transition_dict[targ_stat][new_targ_stat] = transition_deact_tract
        return target_status_transition_dict
