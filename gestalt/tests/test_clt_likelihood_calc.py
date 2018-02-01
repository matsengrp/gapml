import unittest

import numpy as np
import scipy.linalg
import tensorflow as tf

from allele_events import AlleleEvents, Event
from indel_sets import TargetTract, Singleton
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_estimator import CLTLassoEstimator
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB

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

    def _create_model(self, topology, branch_len):
        sess = tf.InteractiveSession()
        num_nodes = len([n for n in topology.traverse("postorder")])
        model = CLTLikelihoodModel(
                topology,
                self.bcode_metadata,
                sess,
                branch_lens = [branch_len for _ in range(num_nodes)],
                target_lams = np.ones(self.num_targets),
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

        branch_len = 10
        model = self._create_model(topology, branch_len)

        # Calculate the hazard using the CLT estimator
        lik_calculator = CLTLassoEstimator(0, model, self.exact_approximator)
        log_lik = lik_calculator.get_log_likelihood(model)

        # Manually calculate the hazard
        hazard_away = model.get_hazard_away([])
        q_mat = np.matrix([[-hazard_away, hazard_away], [0, 0]])
        prob_mat = scipy.linalg.expm(q_mat * branch_len)
        manual_log_prob = np.log(prob_mat[0,0])

        # Check the two are equal
        self.assertEqual(log_lik, manual_log_prob)

        # Calculate the hazard using approximate algo -- should be same
        lik_approx_calc = CLTLassoEstimator(0, model, self.approximator)
        log_lik_approx = lik_approx_calc.get_log_likelihood(model)
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
        model = self._create_model(topology, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        # Calculate the hazard using the CLT estimator
        lik_calculator = CLTLassoEstimator(0, model, self.exact_approximator)
        log_lik = lik_calculator.get_log_likelihood(model)

        # Manually calculate the hazard
        hazard_away = model.get_hazard_away([])
        hazard_to_event = target_lams[0] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        hazard_away_from_event = model.get_hazard_away([TargetTract(0,0,0,0)])
        q_mat = np.matrix([
            [-hazard_away, hazard_to_event, hazard_away - hazard_to_event],
            [0, -hazard_away_from_event, hazard_away_from_event],
            [0, 0, 0]])
        prob_mat = scipy.linalg.expm(q_mat * branch_len)
        manual_cut_log_prob = np.log(prob_mat[0,1])

        # Get prob of deletion - one left, two right
        manual_trim_probs = model.initialize_indel_cond_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = list(manual_trim_probs.values())[0]
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

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
        model = self._create_model(topology, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        # Calculate the hazard using the CLT estimator
        lik_calculator = CLTLassoEstimator(0, model, self.exact_approximator)
        log_lik = lik_calculator.get_log_likelihood(model)

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
        prob_mat = scipy.linalg.expm(q_mat * branch_len)
        manual_cut_log_prob = np.log(prob_mat[0,2])

        # Get prob of deletion
        manual_trim_probs = model.initialize_indel_cond_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = list(manual_trim_probs.values())[0]
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

        # Calculate the hazard using approximate algo -- should be smaller
        lik_approx_calc = CLTLassoEstimator(0, model, self.approximator)
        log_lik_approx = lik_approx_calc.get_log_likelihood(model)
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
        model = self._create_model(topology, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        # Calculate the hazard using the CLT estimator
        lik_calculator = CLTLassoEstimator(0, model, self.exact_approximator)
        log_lik = lik_calculator.get_log_likelihood(model)

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
        prob_mat = scipy.linalg.expm(q_mat * branch_len * 2)
        manual_cut_log_prob = np.log(prob_mat[0,2])

        # Get prob of deletion
        manual_trim_probs = model.initialize_indel_cond_probs([Singleton(
            event.start_pos,
            event.del_len,
            event.min_target,
            event.min_target,
            event.max_target,
            event.max_target,
            event.insert_str)])
        manual_trim_probs = list(manual_trim_probs.values())[0]
        manual_log_prob = manual_cut_log_prob + np.log(manual_trim_probs)

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

        # Calculate the hazard using approximate algo -- should be smaller
        lik_approx_calc = CLTLassoEstimator(0, model, self.approximator)
        log_lik_approx = lik_approx_calc.get_log_likelihood(model)
        self.assertTrue(log_lik_approx < manual_log_prob)
