import unittest

from indel_sets import *
from approximator import ApproximatorLB

class ApproximatorLBTestCase(unittest.TestCase):
    def test_deactivated_targets(self):
        tt_grp = (TargetTract(0,0,1,2), TargetTract(8,8,8,8))
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set([0,1,2,8]))

        tt_grp = ()
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set())

        tt_grp = (TargetTract(0,1,8,9),)
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set(range(10)))

    def test_active_any_trim_targets(self):
        tt_grp = (TargetTract(1,1,1,2), TargetTract(4,4,4,4))
        indel_set = Wildcard(0,8)
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [0,3,5,6,7,8])

        tt_grp = (TargetTract(2,2,3,3), TargetTract(4,4,4,4))
        indel_set = Wildcard(0,5)
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [0,1,5])

        tt_grp = (TargetTract(2,2,3,3),)
        indel_set = SingletonWC(30, 20, 2, 2, 3, 3, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [])

        tt_grp = (TargetTract(2,2,2,2),)
        indel_set = SingletonWC(30, 20, 2, 2, 3, 3, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [])

        tt_grp = ()
        indel_set = SingletonWC(30, 20, 2, 2, 4, 4, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [3])

    def test_partition(self):
        tts = (TargetTract(8,8,8,8),)
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,8)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,8)], (tts[0],))

        tts = (
            TargetTract(1,1,2,2),
            TargetTract(3,3,3,4),
            TargetTract(8,8,8,8))
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], (tts[0], tts[1]))
        self.assertEqual(parts[Wildcard(8,9)], (tts[2],))

        tts = ()
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,9)], ())

    def test_transition_dict_row(self):
        node1 = (TargetTract(3,3,3,3),)
        node2 = (TargetTract(2,2,4,4),)
        node3 = (TargetTract(1,1,5,5),)
        node4 = (TargetTract(0,0,9,9),)
        graph = TransitionGraph()
        graph.add_node(())
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_node(node4)
        graph.add_edge((), TransitionToNode(node1[0], node1))
        graph.add_edge((), TransitionToNode(node2[0], node2))
        graph.add_edge((), TransitionToNode(node3[0], node3))
        graph.add_edge(node2, TransitionToNode(node3[0], node3))
        graph.add_edge(node3, TransitionToNode(node4[0], node4))
        tts_partition_info = {
            node4: {
                "start": (),
                "graph": graph
                }}
        indel_set_list = [node4]
        transition_dict = dict()
        self.mdl._add_transition_dict_row(tts_partition_info, indel_set_list, transition_dict)
        self.assertTrue(() in transition_dict[()])
        self.assertTrue(node1 in transition_dict[()])
        self.assertTrue(node2 in transition_dict[()])
        self.assertTrue(node3 in transition_dict[()])
        self.assertEqual(len(transition_dict[()].keys()), 5)
        self.assertTrue(node3 in transition_dict[node2])
        self.assertTrue(node2 in transition_dict[node2])
        self.assertTrue("unlikely" in transition_dict[node2])
        self.assertEqual(len(transition_dict[node2].keys()), 3)
        self.assertTrue(node4 in transition_dict[node3])
        self.assertEqual(len(transition_dict[node3].keys()), 3)
        self.assertEqual(transition_dict[node4]["unlikely"], 0)
        self.assertEqual(transition_dict[node4][node4], 0)
        self.assertEqual(len(transition_dict[node4].keys()), 2)
