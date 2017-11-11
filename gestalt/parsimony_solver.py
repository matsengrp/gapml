from typing import List
import numpy as np

from cell_lineage_tree import CellLineageTree
from barcode_events import Event, EventWildcard, UnresolvedEvents, BarcodeEvents

class ParsimonySolver:
    """
    Our in-built parsimony engine
    """
    def __init__(self, cut_sites: List[int]):
        self.cut_sites = cut_sites

    def annotate_parsimony_states(self, tree: CellLineageTree):
        """
        get the most parsimonious states for each node given the topology
        """
        print(tree)
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.add_feature("parsimony_barcode_events", node.barcode_events)
                node.add_feature("fun", [str(i) for i in node.barcode_events.events])
            elif node.is_root():
                print(node.get_ascii(attributes=["fun"], show_internal=True))
                # do something
                1/0
            else:
                pars_bcode_evts = self.get_parsimony_barcode(
                    [c.parsimony_barcode_events for c in node.get_children()]
                )
                node.add_feature("parsimony_barcode_events", pars_bcode_evts)
                node.add_feature("fun", [str(i) for i in pars_bcode_evts.events])
                # TODO: propagate anything down if needed


    def get_parsimony_barcode(self, child_bcodes: List[BarcodeEvents]):
        intersect_bcode = child_bcodes[0]
        for c in child_bcodes[1:]:
            intersect_bcode = self._get_parsimony_barcode(intersect_bcode, c)
        return intersect_bcode


    def _cycle_through_evts(self, b1_events: List[UnresolvedEvents], b2_events: List[UnresolvedEvents], FUNCTION):
        parsimony_events = []
        idx1 = 0
        idx2 = 0
        n1 = len(b1_events)
        n2 = len(b2_events)
        while idx1 < n1 and idx2 < n2:
            e1 = b1_events[idx1]
            e2 = b2_events[idx2]

            # not overlapping
            # progress whichever barcode's end pos is behind
            if e1.start_end[0] > e2.start_end[1]:
                idx2 += 1
                continue
            elif e2.start_end[0] > e1.start_end[1]:
                idx1 += 1
                continue

            # overlapping
            use_in_ancestral, pars_evt, idxs_explained = FUNCTION(e1, e2)
            if use_in_ancestral:
                parsimony_events.append(pars_evt)
                # throw away the true events. dont throw away wildcards
                if 0 in idxs_explained:
                    b1_events = b1_events[:idx1] + b1_events[idx1 + 1:]
                    n1 -= 1
                if 1 in idxs_explained:
                    b2_events = b2_events[:idx2] + b2_events[idx2 + 1:]
                    n2 -= 1
                if len(idxs_explained) == 1:
                    # TODO: track the resolved wildcard
                    print("do something")
            else:
                if e1.start_end[1] > e2.start_end[1]:
                    idx2 += 1
                else:
                    idx1 += 1
        return parsimony_events, b1_events, b2_events



    def _cycle_through_evts_one(self, b1_events: List[UnresolvedEvents], b2_events: List[UnresolvedEvents]):
        parsimony_events = []
        idx1 = 0
        idx2 = 0
        n1 = len(b1_events)
        n2 = len(b2_events)
        e1_semi_explained = False
        e2_semi_explained = False
        targets_explained = []
        # By checking in this order, if we have two events that could be explained by the same event
        # but the same event cannot explain both events at once, we will choose the first nested event
        # TODO: maybe we will change this one day. for now, it's not too bad since i think it's really
        # unlikely?
        while idx1 < n1 and idx2 < n2:
            e1 = b1_events[idx1]
            e2 = b2_events[idx2]

            # not overlapping
            # progress whichever barcode's end pos is behind
            if e1.start_end[0] > e2.start_end[1]:
                idx2 += 1
                continue
            elif e2.start_end[0] > e1.start_end[1]:
                idx1 += 1
                continue

            # overlapping
            use_in_ancestral, pars_evt, idxs_explained, targets_explained = self._is_one_score(
                    e1, e2, e1_semi_explained, e2_semi_explained, targets_explained)
            if use_in_ancestral:
                parsimony_events.append(pars_evt)
                if 0 in idxs_explained:
                    b1_events = b1_events[:idx1] + b1_events[idx1 + 1:]
                    e2_semi_explained = True
                    e1_semi_explained = False
                    n1 -= 1
                if 1 in idxs_explained:
                    b2_events = b2_events[:idx2] + b2_events[idx2 + 1:]
                    e1_semi_explained = True
                    e2_semi_explained = False
                    n2 -= 1
            else:
                if e1.start_end[1] > e2.start_end[1]:
                    if e2_semi_explained:
                        b2_events = b2_events[:idx2] + b2_events[idx2 + 1:]
                        e2_semi_explained = False
                    else:
                        idx2 += 1
                else:
                    if e1_semi_explained:
                        b1_events = b1_events[:idx1] + b1_events[idx1 + 1:]
                        e1_semi_explained = False
                    else:
                        idx1 += 1
        return parsimony_events, b1_events, b2_events


    def _get_parsimony_barcode(self, b1: BarcodeEvents, b2: BarcodeEvents):
        b1_events = list(b1.events)
        b2_events = list(b2.events)
        all_parsimony_zero_evts, b1_events, b2_events = self._cycle_through_evts(
                b1_events, b2_events, self._is_zero_score)

        all_parsimony_one_evts, b1_events, b2_events = self._cycle_through_evts_one(
                b1_events, b2_events)

        all_parsimony_two_evts, b1_events, b2_events = self._cycle_through_evts(
                b1_events, b2_events, self._is_two_score)

        all_parsimony_evts = all_parsimony_zero_evts + all_parsimony_one_evts + all_parsimony_two_evts
        return BarcodeEvents(all_parsimony_evts)


    def _is_zero_score(self, evt1: UnresolvedEvents, evt2: UnresolvedEvents):
        if evt1.event and evt2.event:
            if evt1.event == evt2.event:
                return True, evt1, (0, 1)
        elif evt1.event and evt2.wildcard:
            if evt2.wildcard.is_compatible(evt1.event):
                return True, evt1, (0,)
        elif evt2.event and evt1.wildcard:
            if evt1.wildcard.is_compatible(evt2.event):
                return True, evt1, (1,)
        return False, None, None


    def _is_nested(self, nester_evt: Event, nestee_evt: Event):
        """
        @returns whether nestee_evt is completely nested inside nester_evt
        """
        return (nester_evt.start_pos <= nestee_evt.start_pos
            and nestee_evt.del_end <= nester_evt.del_end
            and (
                nester_evt.min_target < nestee_evt.min_target
                or nestee_evt.max_target < nester_evt.max_target
        ))


    def _is_one_score(self, evt1: UnresolvedEvents, evt2: UnresolvedEvents, e1_semi_explained, e2_semi_explained, old_targets_explained):
        if evt1.event and evt2.event:
            potential_targets_explained = list(range(
                max(evt1.event.min_target, evt2.event.min_target),
                min(evt1.event.max_target, evt2.event.max_target) + 1))
            # You cannot explain the more than one event by using all the targets
            if e1_semi_explained:
                if len(potential_targets_explained + old_targets_explained) == evt1.event.max_target - evt1.event.min_target + 1:
                    return False, None, None, []
            if e2_semi_explained:
                if len(potential_targets_explained + old_targets_explained) == evt2.event.max_target - evt2.event.min_target + 1:
                    return False, None, None, []
            if self._is_nested(evt1.event, evt2.event):
                assert(not e2_semi_explained)
                return True, evt2, (1,), old_targets_explained + potential_targets_explained
            elif self._is_nested(evt2.event, evt1.event):
                assert(not e1_semi_explained)
                return True, evt1, (0,), old_targets_explained + potential_targets_explained
        return False, None, None, []


    def _is_two_score(self, evt1: UnresolvedEvents, evt2: UnresolvedEvents):
        event1_first = evt1.del_end < evt2.del_end

        start_pos = max(evt1.start_pos, evt2.start_pos)
        start_same = evt1.start_pos == evt2.start_pos
        del_end = min(evt1.del_end, evt2.del_end)
        end_same = evt1.del_end == evt2.del_end

        min_target = max(evt1.min_target, evt2.min_target)
        if evt1.min_target == evt2.min_target and not start_same:
            start_pos = self.cut_sites[evt1.min_target] + 1
            min_target += 1

        max_target = min(evt1.max_target, evt2.max_target)
        if evt1.max_target == evt2.max_target and not end_same:
            del_end = self.cut_sites[evt1.max_target] - 1
            max_target -= 1

        if start_same and end_same:
            # then the insert string must have been different
            # we need to chop off one of the edges. we're going to do it at random
            # TODO: maybe there is a better way in the future. but we really dont want
            # to deal with too many possibilities in the parsimony tree.
            # it's probably fine since this is really unlikely.
            coin_flip = np.random.binomial(1, p=.5)
            if coin_flip == 0:
                min_target += 1
                start_pos = self.cut_sites[evt1.min_target] + 1

        if min_target > max_target or del_end <= start_pos:
            return False, None, None
        wildcard = EventWildcard(
            start_pos,
            del_end - start_pos,
            min_target,
            max_target)
        return True, UnresolvedEvents(wildcard=wildcard), (0 if event1_first else 1,)
