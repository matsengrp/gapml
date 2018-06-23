import unittest

import collapsed_tree
from cell_lineage_tree import CellLineageTree
from allele_events import AlleleEvents, Event

class CollapsedTreeTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10

    def test_ultrametric_simple(self):
        tree = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        child1 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 1)
        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)],
                dist = 2)
        tree.add_child(child1)
        child1.add_child(child2)
        max_dist = child1.dist + child2.dist

        coll_tree = collapsed_tree.collapse_ultrametric(tree)
        
        for leaf in coll_tree:
            self.assertEqual(leaf.get_distance(coll_tree), max_dist)

        num_nodes = 0
        for node in coll_tree.traverse():
            num_nodes += 1
        self.assertEqual(num_nodes, 2)

    def test_ultrametric_branching(self):
        tree = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        child1 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 1)
        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)],
                dist = 2)
        child3 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 2)
        child4 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 3)
        tree.add_child(child1)
        tree.add_child(child4)
        child1.add_child(child2)
        child1.add_child(child3)
        max_dist = child1.dist + child2.dist

        coll_tree = collapsed_tree.collapse_ultrametric(tree)
        
        for leaf in coll_tree:
            self.assertEqual(leaf.get_distance(coll_tree), max_dist)

        num_nodes = 0
        for node in coll_tree.traverse():
            num_nodes += 1
        self.assertEqual(num_nodes, 4)

        self.assertEqual(
            len(set([leaf.allele_events_list_str for leaf in coll_tree])),
            2)

    def test_ultrametric_many_branching(self):
        tree = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        child1 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 1)
        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)],
                dist = 2)
        child3 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 2)
        child4 = CellLineageTree(
                allele_events_list = [AlleleEvents(num_targets=self.num_targets)],
                dist = 2)
        event5 = Event(
                start_pos = 2,
                del_len = 2,
                min_target = 0,
                max_target = 0,
                insert_str = "wee!")
        child5 = CellLineageTree(
                allele_events_list=[AlleleEvents([event5], num_targets=self.num_targets)],
                dist = 0.5)
        event6 = Event(
                start_pos = 20,
                del_len = 20,
                min_target = 1,
                max_target = 1,
                insert_str = "foo")
        child6 = CellLineageTree(
                allele_events_list=[AlleleEvents([event5, event6], num_targets=self.num_targets)],
                dist = 0.5)
        child7 = CellLineageTree(
                allele_events_list=[AlleleEvents([event5], num_targets=self.num_targets)],
                dist = 0.5)
        tree.add_child(child1)
        tree.add_child(child4)
        child1.add_child(child2)
        child1.add_child(child3)
        child4.add_child(child5)
        child5.add_child(child6)
        child5.add_child(child7)
        max_dist = child1.dist + child2.dist

        coll_tree = collapsed_tree.collapse_ultrametric(tree)
        
        for leaf in coll_tree:
            self.assertEqual(leaf.get_distance(coll_tree), max_dist)

        num_nodes = 0
        for node in coll_tree.traverse():
            num_nodes += 1
        self.assertEqual(num_nodes, 8)

        self.assertEqual(
            len(set([leaf.allele_events_list_str for leaf in coll_tree])),
            4)
