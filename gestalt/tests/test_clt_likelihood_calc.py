import unittest

import numpy as np
import scipy.linalg
import tensorflow as tf

import ancestral_events_finder as anc_evt_finder
from allele_events import AlleleEvents, Event
from indel_sets import TargetTract, Singleton
from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB
import tf_common

class LikelihoodCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_metadata = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_sites = [3, 3, 3],
            crucial_pos_len = [3,3])
        self.num_targets = self.bcode_metadata.n_targets

        # Don't self.approximate -- exact calculation
        self.exact_approximator = ApproximatorLB(extra_steps=10, anc_generations=3, bcode_metadata=self.bcode_metadata)
        # Actually do an approximation
        self.approximator = ApproximatorLB(extra_steps=1, anc_generations=1, bcode_metadata=self.bcode_metadata)

    def _create_model(self, topology, bcode_metadata, branch_len):
        sess = tf.InteractiveSession()
        num_nodes = len([n for n in topology.traverse("postorder")])
        model = CLTLikelihoodModel(
                topology,
                bcode_metadata,
                sess,
                branch_lens = np.array([branch_len for _ in range(num_nodes)]),
                target_lams = np.ones(bcode_metadata.n_targets) + np.random.uniform(
                    size=bcode_metadata.n_targets) * 0.1,
                trim_long_probs = 0.1 * np.ones(2),
                trim_zero_prob = 0.5,
                trim_poissons = np.ones(2),
                insert_zero_prob = 0.5,
                insert_poisson = 2)
        tf.global_variables_initializer().run()
        return model

    def test_branch_likelihood_no_events(self):
        # Create a branch with no events
        topology = CellLineageTree()
        child = CellLineageTree(allele_events=AlleleEvents([], num_targets=self.num_targets))
        topology.add_child(child)

        branch_len = 0.1
        model = self._create_model(topology, self.bcode_metadata, branch_len)

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_nodes = model._create_hazard_away_nodes([()])
        hazard_aways = model.sess.run(hazard_away_nodes)
        hazard_away = hazard_aways[0]
        q_mat = np.matrix([[-hazard_away, hazard_away], [0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len)[0]
        manual_log_prob = np.log(prob_mat[0,0])

        # Check the two are equal
        self.assertEqual(log_lik, manual_log_prob)

        # Calculate the hazard using approximate algo -- should be same
        t_mats = self.approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()
        self.assertEqual(log_lik_approx, manual_log_prob)

    def test_branch_likelihood(self):
        # Create a branch with one event
        topology = CellLineageTree()
        event = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child = CellLineageTree(
                allele_events=AlleleEvents([event], num_targets=self.num_targets))
        topology.add_child(child)

        branch_len = 10
        model = self._create_model(topology, self.bcode_metadata, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_nodes = model._create_hazard_away_nodes([(), (TargetTract(0,0,0,0),)])
        hazard_aways = model.sess.run(hazard_away_nodes)
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
        topology = CellLineageTree()
        event = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "ATC")
        child = CellLineageTree(
                allele_events=AlleleEvents([event], num_targets=self.num_targets))
        topology.add_child(child)

        branch_len = 10
        model = self._create_model(topology, self.bcode_metadata, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard -- can cut the middle only or cut the whole barcode
        hazard_away = model.get_hazard_away([])
        hazard_to_cut1 = target_lams[1] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_to_cut03 = target_lams[0] * target_lams[2] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_away_from_cut1 = model.get_hazard_away([TargetTract(1,1,1,1)])

        q_mat = np.matrix([
            [-hazard_away, hazard_to_cut1, hazard_to_cut03, hazard_away - hazard_to_cut1 - hazard_to_cut03],
            [0, -hazard_away_from_cut1, hazard_to_cut03, hazard_away - hazard_to_cut03],
            [0, 0, 0, 0],
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

        # Calculate the hazard using approximate algo -- should be smaller
        t_mats = self.approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()
        self.assertTrue(log_lik_approx < manual_log_prob)

    def test_two_branch_likelihood_big_intertarget_del(self):
        # Create two branches: Root -- child1 -- child2
        topology = CellLineageTree()
        event = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "ATC")
        child1 = CellLineageTree(
                allele_events=AlleleEvents([], num_targets=self.num_targets))
        child2 = CellLineageTree(
                allele_events=AlleleEvents([event], num_targets=self.num_targets))
        topology.add_child(child1)
        child1.add_child(child2)

        branch_len = 10
        model = self._create_model(topology, self.bcode_metadata, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # The probability should be the same as a single branch with double the length
        # Manually calculate the hazard -- can cut the middle only or cut the whole barcode
        hazard_away = model.get_hazard_away([])
        hazard_to_cut1 = target_lams[1] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_to_cut03 = target_lams[0] * target_lams[2] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_away_from_cut1 = model.get_hazard_away([TargetTract(1,1,1,1)])

        q_mat = np.matrix([
            [-hazard_away, hazard_to_cut1, hazard_to_cut03, hazard_away - hazard_to_cut1 - hazard_to_cut03],
            [0, -hazard_away_from_cut1, hazard_to_cut03, hazard_away - hazard_to_cut03],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len * 2)[0]
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

        # Calculate the hazard using approximate algo -- should be smaller
        t_mats = self.approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()
        self.assertTrue(log_lik_approx < manual_log_prob)

    def test_three_branch_likelihood(self):
        bcode_metadata = BarcodeMetadata()
        num_targets = bcode_metadata.n_targets

        # Don't self.approximate -- exact calculation
        exact_approximator = ApproximatorLB(extra_steps=10, anc_generations=5, bcode_metadata=bcode_metadata)
        # Actually do an approximation
        approximator = ApproximatorLB(extra_steps=1, anc_generations=1, bcode_metadata=bcode_metadata)

        # Create three branches: Root -- child1 -- child2 -- child3
        topology = CellLineageTree()
        event1 = Event(
                start_pos = 6,
                del_len = 23,
                min_target = 0,
                max_target = 0,
                insert_str = "ATC")
        event2 = Event(
                start_pos = 23 + 20,
                del_len = 20 + 23 * 2,
                min_target = 1,
                max_target = 3,
                insert_str = "")
        child1 = CellLineageTree(
                allele_events=AlleleEvents([], num_targets=num_targets))
        child2 = CellLineageTree(
                allele_events=AlleleEvents([], num_targets=num_targets))
        child3 = CellLineageTree(
                allele_events=AlleleEvents([event1, event2], num_targets=num_targets))
        topology.add_child(child1)
        child1.add_child(child2)
        child2.add_child(child3)

        branch_len = 0.5
        model = self._create_model(topology, bcode_metadata, branch_len)

        t_mats = exact_approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Calculate the hazard using approximate algo -- should be smaller
        t_mats = approximator.create_transition_matrix_wrappers(model)
        model.create_topology_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()

        print("log lik approx", log_lik_approx)
