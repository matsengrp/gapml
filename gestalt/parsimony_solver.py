from typing import List
import numpy as np

from cell_lineage_tree import CellLineageTree
from barcode_events import Event, BarcodeEvents

class MaxEventSolver:
    """
    Our in-built "parsimony" engine
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

    def _get_parsimony_barcode(parent_bcode: BarcodeEvents, child_bcode: BarcodeEvents):
        child_idx = 0
        child_evt = child_bcode.events[child_idx]
        new_parent_events = []
        for par_evt in parent_bcode.events:
            while child_evt.del_end < par_evt.start_pos:
                child_idx += 1
                if child_idx == len(child_bcode.events):
                    # we're done with cycling through child, so we're done
                    return new_parent_events
                child_evt = child_bcode.events[child_idx]

            # Check if events overlap
            if par_evt.del_end < child_evt.start_pos:
                continue
            else:
                # Now we have overlapping events
                
                # Is parent event valid?
                if par_evt.generalized_equals(child_evt) or self._is_nested(par_evt, child_evt):
                    new_parent_events.append(par_evt)
                else:
                    # No parent is not valid. Make new parent
                    new_par = self._make_new_parent(par_evt, child_evt)
                    new_parent_events.append(new_par)

        return BarcodeEvents(new_parent_events)

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


    def _make_new_parent(self, evt1: Event, evt2: Event):
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
            return None
        else:
            return Event(
                start_pos,
                del_end - start_pos,
                "*",
                min_target,
                max_target)
