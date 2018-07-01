import unittest

import numpy as np
import scipy.linalg
import tensorflow as tf

import tf_common
from allele_events import AlleleEvents, Event
from indel_sets import TargetTract, Singleton
from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from collapsed_tree import collapse_zero_lens
from transition_wrapper_maker import TransitionWrapperMaker
from target_status import TargetStatus, TargetDeactTract

class LikelihoodCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_metadata = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_site = 3,
            crucial_pos_len = [3,3])
        self.num_targets = self.bcode_metadata.n_targets

    def _create_bifurc_model(
            self,
            topology,
            bcode_metadata,
            branch_len,
            branch_lens = [],
            double_cut_weight = 0.3,
            target_lams = None):
        sess = tf.InteractiveSession()
        num_nodes = topology.get_num_nodes()
        if len(branch_lens) == 0:
            branch_lens = [branch_len for _ in range(num_nodes)]
        else:
            branch_lens = [0] + [br_len for br_len in branch_lens]

        if not hasattr(topology, "node_id"):
            topology.label_node_ids()
        for node in topology.traverse():
            node.dist = branch_lens[node.node_id]
            if node.is_leaf():
                tot_time = node.get_distance(topology)

        if target_lams is None:
            target_lams = np.ones(bcode_metadata.n_targets) + np.random.uniform(size=bcode_metadata.n_targets) * 0.1

        model = CLTLikelihoodModel(
                topology,
                bcode_metadata,
                sess,
                branch_len_inners = np.array(branch_lens),
                branch_len_offsets_proportion = 1e-20 * np.ones(num_nodes), # dummy!!!
                target_lams = target_lams,
                trim_long_probs = 0.1 * np.ones(2),
                trim_zero_probs = 0.5 * np.ones(2),
                trim_short_poissons = np.ones(2),
                trim_long_poissons = np.ones(2),
                insert_zero_prob = 0.5,
                insert_poisson = 2,
                double_cut_weight = double_cut_weight,
                tot_time = tot_time)
        tf.global_variables_initializer().run()
        return model

    def _create_multifurc_model(self,
            topology,
            bcode_metadata,
            branch_len_inners,
            branch_len_offsets,
            tot_time = 1,
            double_cut_weight = 0.3,
            target_lams = None):
        sess = tf.InteractiveSession()
        num_nodes = topology.get_num_nodes()

        if target_lams is None:
            target_lams = np.ones(bcode_metadata.n_targets) + np.random.uniform(size=bcode_metadata.n_targets) * 0.1
        br_len_inners = np.array(branch_len_inners, dtype=float)
        br_len_offsets = np.array(branch_len_offsets, dtype=float)
        model = CLTLikelihoodModel(
                topology,
                bcode_metadata,
                sess,
                branch_len_inners = br_len_inners,
                branch_len_offsets_proportion = br_len_offsets/(br_len_inners + 1e-10),
                target_lams = target_lams,
                trim_long_probs = 0.1 * np.ones(2),
                trim_zero_probs = 0.5 * np.ones(2),
                trim_short_poissons = np.ones(2),
                trim_long_poissons = np.ones(2),
                insert_zero_prob = 0.5,
                insert_poisson = 2,
                double_cut_weight = double_cut_weight,
                tot_time = tot_time)
        tf.global_variables_initializer().run()
        return model, target_lams

    def test_multifurcation_resolution(self):
        # Create multifurcating tree -- this is star tree
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)])
        topology.add_child(child2)
        child2.add_feature("node_id", 2)

        event3 = Event(
                start_pos = 6,
                del_len = 10,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event3], num_targets=self.num_targets)])
        topology.add_child(child3)
        child3.add_feature("node_id", 3)

        tot_time = 4
        branch_len_inners = [0, 4, 4, 4]
        branch_len_offsets = [0, 3, 2, 3]
        model, target_lams = self._create_multifurc_model(
                topology,
                self.bcode_metadata,
                branch_len_inners,
                branch_len_offsets,
                tot_time = tot_time)
        trim_long_probs = model.trim_long_probs.eval()

        transition_maker = TransitionWrapperMaker(topology, bcode_metadata=self.bcode_metadata)
        model.create_log_lik(transition_maker.create_transition_wrappers())
        multifurc_log_lik, _ = model.get_log_lik()

        # Create equivalent bifurcating tree
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        topology_ext1 = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(topology_ext1)
        topology_ext1.add_feature("node_id", 1)
        topology_ext1.dist = 2

        topology_ext2 = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology_ext1.add_child(topology_ext2)
        topology_ext2.add_feature("node_id", 2)
        topology_ext2.dist = 1

        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)])
        topology_ext1.add_child(child2)
        child2.add_feature("node_id", 3)
        child2.dist = 2

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology_ext2.add_child(child1)
        child1.add_feature("node_id", 4)
        child1.dist = 1

        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event3], num_targets=self.num_targets)])
        topology_ext2.add_child(child3)
        child3.add_feature("node_id", 5)
        child3.dist = 1

        branch_lens = [2, 1, 2, 1, 1]
        bifurc_model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len=0,
                branch_lens=branch_lens,
                target_lams=target_lams)
        transition_maker = TransitionWrapperMaker(topology, bcode_metadata=self.bcode_metadata)
        bifurc_model.create_log_lik(transition_maker.create_transition_wrappers())
        bifurc_log_lik, _ = bifurc_model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_dict = bifurc_model._create_hazard_away_dict()
        hazard_aways = bifurc_model.sess.run([
            hazard_away_dict[TargetStatus()],
            hazard_away_dict[TargetStatus(TargetDeactTract(0,0))]])
        hazard_away = hazard_aways[0]
        hazard_away_from_event = hazard_aways[1]
        hazard_to_event = target_lams[0] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        q_mat = np.matrix([
            [-hazard_away, hazard_to_event, hazard_away - hazard_to_event],
            [0, -hazard_away_from_event, hazard_away_from_event],
            [0, 0, 0]])
        manual_cut_log_prob = -hazard_away * tot_time
        for idx in range(2,4):
            prob_mat = tf_common._custom_expm(q_mat, branch_lens[idx])[0]
            manual_cut_log_prob += np.log(prob_mat[0,1])
        # Get prob of deletion - one left, two right
        manual_trim_log_prob2 = bifurc_model._create_log_indel_probs([Singleton(
            event2.start_pos,
            event2.del_len,
            event2.min_target,
            event2.min_target,
            event2.max_target,
            event2.max_target,
            event2.insert_str)])
        manual_trim_log_prob3 = bifurc_model._create_log_indel_probs([Singleton(
            event3.start_pos,
            event3.del_len,
            event3.min_target,
            event3.min_target,
            event3.max_target,
            event3.max_target,
            event3.insert_str)])
        log_trim_probs = bifurc_model.sess.run([manual_trim_log_prob2, manual_trim_log_prob3])
        manual_log_prob = manual_cut_log_prob + np.sum(log_trim_probs)

        # All three calculations should match
        self.assertTrue(np.isclose(multifurc_log_lik, manual_log_prob))
        self.assertTrue(np.isclose(bifurc_log_lik, manual_log_prob))

    def test_multifurc_vs_bifurc(self):
        """
        Consider a more complex multifurcation/bifurcating tree
        Check that hte log likelihoods are the same
        """
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        event0 = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "atc")
        child0 = CellLineageTree(allele_events_list=[AlleleEvents([event0], num_targets=self.num_targets)])
        topology.add_child(child0)
        child0.add_feature("node_id", 1)
        child0.dist = 3

        event1 = Event(
                start_pos = 16,
                del_len = 3,
                min_target = 1,
                max_target = 1,
                insert_str = "")
        child1 = CellLineageTree(allele_events_list=[AlleleEvents([event1], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 2)
        child1.dist = 1

        child2 = CellLineageTree(allele_events_list=[AlleleEvents([event1], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 3)
        child2.dist = 1

        childcopy = CellLineageTree(allele_events_list = [AlleleEvents([event1], num_targets=self.num_targets)])
        child2.add_child(childcopy)
        childcopy.add_feature("node_id", 4)
        childcopy.dist = 1

        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event1, event2], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 5)
        child3.dist = 2

        event3 = Event(
                start_pos = 6,
                del_len = 10,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child4 = CellLineageTree(
                allele_events_list=[AlleleEvents([event1, event3], num_targets=self.num_targets)])
        child2.add_child(child4)
        child4.add_feature("node_id", 6)
        child4.dist = 1

        target_lams = (
                np.ones(self.bcode_metadata.n_targets)
                + np.random.uniform(size=self.bcode_metadata.n_targets) * 0.1)
        branch_lens = [3, 1, 1, 1, 2, 1]
        bifurc_model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len=0,
                branch_lens=branch_lens,
                target_lams=target_lams)
        transition_maker = TransitionWrapperMaker(topology, self.bcode_metadata)
        bifurc_model.create_log_lik(
                transition_maker.create_transition_wrappers())
        bifurc_log_lik, _ = bifurc_model.get_log_lik()

        ####
        # Multifurcating tree calculation
        ####
        # Create collapsed tree
        # Get dist to earliest node with same ancestral state
        multifurc_tree = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        multifurc_tree.add_feature("node_id", 0)

        child0 = CellLineageTree(allele_events_list=[AlleleEvents([event0], num_targets=self.num_targets)])
        multifurc_tree.add_child(child0)
        child0.add_feature("node_id", 1)
        child0.dist = 3

        child1 = CellLineageTree(allele_events_list=[AlleleEvents([event1], num_targets=self.num_targets)])
        multifurc_tree.add_child(child1)
        child1.add_feature("node_id", 2)
        child1.dist = 1

        child2 = CellLineageTree(allele_events_list=[AlleleEvents([event1], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 3)
        child2.dist = 2

        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event1, event2], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 4)
        child3.dist = 2

        child4 = CellLineageTree(
                allele_events_list=[AlleleEvents([event1, event3], num_targets=self.num_targets)])
        child1.add_child(child4)
        child4.add_feature("node_id", 5)
        child4.dist = 2

        tot_time = 3
        branch_len_inners = [0] * 6
        for node in multifurc_tree.traverse():
            branch_len_inners[node.node_id] = node.dist
        branch_len_offsets = [0, 0, 0, 1, 0, 1]

        multifurc_model, _  = self._create_multifurc_model(
                multifurc_tree,
                self.bcode_metadata,
                branch_len_inners,
                branch_len_offsets,
                target_lams = target_lams,
                tot_time = tot_time)
        resolved_bifurc_tree = multifurc_model.get_fitted_bifurcating_tree()
        transition_maker = TransitionWrapperMaker(multifurc_tree, self.bcode_metadata)
        multifurc_model.create_log_lik(
                transition_maker.create_transition_wrappers())
        multifurc_log_lik, _ = multifurc_model.get_log_lik()
        self.assertTrue(np.isclose(multifurc_log_lik, bifurc_log_lik))

    def test_branch_likelihood_no_events(self):
        # Create a branch with no events
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        child = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child)

        branch_len = 0.1
        model = self._create_bifurc_model(topology, self.bcode_metadata, branch_len)

        transition_wrappers = TransitionWrapperMaker(topology, self.bcode_metadata).create_transition_wrappers()
        model.create_log_lik(transition_wrappers)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_dict = model._create_hazard_away_dict()
        hazard_away = model.sess.run(hazard_away_dict[TargetStatus()])
        q_mat = np.matrix([[-hazard_away, hazard_away], [0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len)[0]
        manual_log_prob = np.log(prob_mat[0,0])

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

    def test_branch_likelihood(self):
        # Create a branch with one event
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", True)
        event = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child = CellLineageTree(
                allele_events_list=[AlleleEvents([event], num_targets=self.num_targets)])
        topology.add_child(child)
        child.add_feature("observed", True)

        branch_len = 10
        model = self._create_bifurc_model(topology, self.bcode_metadata, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        transition_wrappers = TransitionWrapperMaker(topology, self.bcode_metadata).create_transition_wrappers()
        model.create_log_lik(transition_wrappers)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_dict = model._create_hazard_away_dict()
        hazard_aways = model.sess.run([
            hazard_away_dict[TargetStatus()],
            hazard_away_dict[TargetStatus(TargetDeactTract(0,0))]])
        hazard_away = hazard_aways[0]
        hazard_away_from_event = hazard_aways[1]
        hazard_to_event = target_lams[0] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        q_mat = np.matrix([
            [-hazard_away, hazard_to_event, hazard_away - hazard_to_event],
            [0, -hazard_away_from_event, hazard_away_from_event],
            [0, 0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len)[0]
        manual_cut_log_prob = np.log(prob_mat[0,1])

        # Get prob of deletion - one left, two right
        manual_trim_log_prob_node = model._create_log_indel_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = np.exp(model.sess.run(manual_trim_log_prob_node)[0])
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik, manual_log_prob))

    def test_branch_likelihood_big_intertarget_del(self):
        # Create one branch
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        event = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "ATC")
        child = CellLineageTree(
                allele_events_list=[AlleleEvents([event], num_targets=self.num_targets)])
        topology.add_child(child)

        branch_len = 10
        double_cut_weight  = 0.3
        model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len,
                double_cut_weight = double_cut_weight)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        transition_wrappers = TransitionWrapperMaker(topology, self.bcode_metadata).create_transition_wrappers()
        model.create_log_lik(transition_wrappers)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard -- can cut the middle only or cut the whole barcode
        hazard_away_dict = model._create_hazard_away_dict()
        hazard_away, hazard_away_from_cut1, hazard_away_cut02 = model.sess.run([
            hazard_away_dict[TargetStatus()],
            hazard_away_dict[TargetStatus(TargetDeactTract(1,1))],
            hazard_away_dict[TargetStatus(TargetDeactTract(0,2))],
            ])
        hazard_to_cut1 = target_lams[1] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_to_cut02 = double_cut_weight * target_lams[0] * target_lams[2] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])

        q_mat = np.matrix([
            [-hazard_away, hazard_to_cut1, hazard_to_cut02, hazard_away - hazard_to_cut1 - hazard_to_cut02],
            [0, -hazard_away_from_cut1, hazard_to_cut02, hazard_away_from_cut1 - hazard_to_cut02],
            [0, 0, -hazard_away_cut02, hazard_away_cut02],
            [0, 0, 0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len)[0]
        manual_cut_log_prob = np.log(prob_mat[0,2])

        # Get prob of deletion
        manual_trim_log_prob_node = model._create_log_indel_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = np.exp(model.sess.run(manual_trim_log_prob_node)[0])
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

    def test_two_branch_likelihood_big_intertarget_del(self):
        # Create two branches: Root -- child1 -- child2
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", True)
        event = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "atc")
        child1 = CellLineageTree(
                allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_child(child2)
        child1.add_feature("observed", False)
        child2.add_feature("observed", True)

        branch_len1 = 10
        branch_len2 = 5
        double_cut_weight = 0.3
        model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len=None,
                branch_lens=[branch_len1, branch_len2],
                double_cut_weight = double_cut_weight)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        transition_wrappers = TransitionWrapperMaker(topology, self.bcode_metadata).create_transition_wrappers()
        model.create_log_lik(transition_wrappers)
        log_lik, _ = model.get_log_lik()

        # The probability should be the same as a single branch with double the length
        # Manually calculate the hazard -- can cut the middle only or cut the whole barcode
        hazard_away_dict = model._create_hazard_away_dict()
        hazard_away, hazard_away_from_cut1 = model.sess.run([
            hazard_away_dict[TargetStatus()],
            hazard_away_dict[TargetStatus(TargetDeactTract(1,1))]])
        hazard_to_cut1 = target_lams[1] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_to_cut02 = double_cut_weight * target_lams[0] * target_lams[2] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])

        q_mat = np.matrix([
            [-hazard_away, hazard_to_cut1, hazard_to_cut02, hazard_away - hazard_to_cut1 - hazard_to_cut02],
            [0, -hazard_away_from_cut1, hazard_to_cut02, hazard_away - hazard_to_cut02],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len1 + branch_len2)[0]
        manual_cut_log_prob = np.log(prob_mat[0,2])

        # Get prob of deletion
        manual_trim_log_prob_node = model._create_log_indel_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = np.exp(model.sess.run(manual_trim_log_prob_node)[0])
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))
