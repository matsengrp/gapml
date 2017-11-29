from typing import List, Tuple
import numpy as np

from cell_lineage_tree import CellLineageTree
from barcode_events import Event, BarcodeEvents

class AncestralEventsFinder:
    """
    Our in-built engine for finding the union of all possible
    events in the internal nodes.
    """
    def __init__(self, orig_length: int, active_positions: List[Tuple[int, int]]):
        self.orig_length = orig_length
        self.active_pos = active_positions
        self.n_targets = len(active_positions)

    def annotate_ancestral_events(self, tree: CellLineageTree):
        """
        Find all possible events in the internal nodes.
        (Finds a set of events of which parsimony is a subset, I think)
        """
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.add_feature("reference_ancestral_events", node.barcode_events)
                node.add_feature("fun", [str(i) for i in node.barcode_events.events])
            elif node.is_root():
                print(node.get_ascii(attributes=["fun"], show_internal=True))
                #print(node.get_ascii(attributes=["cell_state"], show_internal=True))
                # do something
                1/0
            else:
                pars_bcode_evts = self.get_possible_parent_events(
                    [c.reference_ancestral_events for c in node.get_children()]
                )
                node.add_feature("reference_ancestral_events", pars_bcode_evts)
                node.add_feature("fun", [str(i) for i in pars_bcode_evts.events])


    def get_possible_parent_events(self, child_bcodes: List[BarcodeEvents]):
        # TODO: is there a python reduce function?
        intersect_bcode = child_bcodes[0]
        for c in child_bcodes[1:]:
            intersect_bcode = self._get_possible_parent_events(intersect_bcode, c)
        return intersect_bcode

    def _get_possible_parent_events(self, parent_bcode: BarcodeEvents, child_bcode: BarcodeEvents):
        """
        @param parent_bcode: barcode to treat as parent
        @param child_bcode: barcode to treat as child
        @return BarcodeEvents with the union of all possible events that could be
                in the parent barcode
        """
        if len(child_bcode.events) == 0:
            return BarcodeEvents()

        child_idx = 0
        par_idx = 0
        num_child_evts = len(child_bcode.events)
        num_par_evts = len(parent_bcode.events)
        new_parent_events = []
        while par_idx < num_par_evts and child_idx < num_child_evts:
            par_evt = parent_bcode.events[par_idx]
            child_evt = child_bcode.events[child_idx]

            if child_evt.del_end < par_evt.start_pos:
                child_idx += 1
                continue
            elif par_evt.del_end < child_evt.start_pos:
                par_idx += 1
                continue

            # Now we have overlapping events
            if par_evt.generalized_equals(child_evt) or self._is_nested(child_evt, par_evt):
                # Is parent event valid?
                new_parent_events.append(par_evt)
            elif self._is_nested(par_evt, child_evt):
                # make child event the parent instead?
                new_parent_events.append(child_evt)
            else:
                # Make a completely new parent event
                new_par = self._make_new_parent(par_evt, child_evt)
                if new_par is not None:
                    new_parent_events.append(new_par)

            if par_evt.del_end > child_evt.del_end:
                child_idx += 1
            else:
                par_idx += 1

        return BarcodeEvents(new_parent_events)

    def _is_nested(self, nester_evt: Event, nestee_evt: Event):
        """
        @returns whether nestee_evt is completely nested inside nester_evt
        """
        if (nester_evt.start_pos <= nestee_evt.start_pos
                and nestee_evt.del_end <= nester_evt.del_end):
            if nester_evt.is_wildcard:
                return (nester_evt.min_target <= nestee_evt.min_target
                        and nestee_evt.max_target <= nester_evt.max_target)
            else:
                return (nester_evt.min_target < nestee_evt.min_target
                    and nestee_evt.max_target < nester_evt.max_target)
        else:
            return False

    def _make_new_parent(self, evt1: Event, evt2: Event):
        """
        Intersect the two events to create a new parent event
        """
        min_target = max(evt1.min_target, evt2.min_target)
        max_target = min(evt1.max_target, evt2.max_target)

        if not (evt1.is_wildcard and evt2.is_wildcard):
            min_target += 1
            max_target -= 1

        # Start pos is the first position where we don't disturb min_target - 1
        start_pos = self.active_pos[min_target - 1][1] if min_target > 0 else 0
        # End pos is the first position where we don't disturb max_target + 1
        del_end = self.active_pos[max_target + 1][0] if max_target < self.n_targets - 1 else self.barcode_length

        if min_target > max_target or del_end <= start_pos:
            return None
        else:
            return Event(
                start_pos,
                del_end - start_pos,
                min_target,
                max_target,
                insert_str="*")
