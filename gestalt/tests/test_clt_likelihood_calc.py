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
from collapsed_tree import collapse_zero_lens

class LikelihoodCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_metadata = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_site = 3,
            crucial_pos_len = [3,3])
        self.num_targets = self.bcode_metadata.n_targets

        # Don't self.approximate -- exact calculation
        self.exact_approximator = ApproximatorLB(extra_steps=10, anc_generations=3, bcode_metadata=self.bcode_metadata)
        # Actually do an approximation
        self.approximator = ApproximatorLB(extra_steps=1, anc_generations=1, bcode_metadata=self.bcode_metadata)

    def _create_bifurc_model(self, topology, bcode_metadata, branch_len, branch_lens = [], target_lams = None):
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
                branch_len_offsets= 1e-20 * np.ones(num_nodes), # dummy!!!
                target_lams = target_lams,
                trim_long_probs = 0.1 * np.ones(2),
                trim_zero_prob = 0.5,
                trim_poissons = np.ones(2),
                insert_zero_prob = 0.5,
                insert_poisson = 2,
                tot_time = tot_time)
        tf.global_variables_initializer().run()
        return model

    def _create_multifurc_model(self,
            topology,
            bcode_metadata,
            branch_len_inners,
            branch_len_offsets,
            tot_time = 1,
            target_lams = None):
        sess = tf.InteractiveSession()
        num_nodes = topology.get_num_nodes()

        if target_lams is None:
            target_lams = np.ones(bcode_metadata.n_targets) + np.random.uniform(size=bcode_metadata.n_targets) * 0.1
        model = CLTLikelihoodModel(
                topology,
                bcode_metadata,
                sess,
                branch_len_inners = np.array(branch_len_inners, dtype=float),
                branch_len_offsets= np.array(branch_len_offsets, dtype=float),
                target_lams = target_lams,
                trim_long_probs = 0.1 * np.ones(2),
                trim_zero_prob = 0.5,
                trim_poissons = np.ones(2),
                insert_zero_prob = 0.5,
                insert_poisson = 2,
                tot_time = tot_time)
        tf.global_variables_initializer().run()
        return model, target_lams

    def test_multifurcation_resolution(self):
        # Create multifurcating tree
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", False)
        topology.add_feature("node_id", 0)
        topology.add_feature("is_copy", False)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("observed", True)
        child1.add_feature("node_id", 1)
        child1.add_feature("is_copy", True)

        event2 = Event(
                start_pos = 6,
                del_len = 3,
                min_target = 0,
                max_target = 0,
                insert_str = "")
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)])
        topology.add_child(child2)
        child2.add_feature("observed", True)
        child2.add_feature("is_copy", False)
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
        child3.add_feature("observed", True)
        child3.add_feature("is_copy", False)
        child3.add_feature("node_id", 3)

        tot_time = 4
        branch_len_inners = [0, 1, 2, 1]
        branch_len_offsets = [0, 3, 2, 3]
        model, target_lams = self._create_multifurc_model(
                topology,
                self.bcode_metadata,
                branch_len_inners,
                branch_len_offsets,
                tot_time = tot_time)
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        multifurc_log_lik, _ = model.get_log_lik()

        # Create equivalent bifurcating tree      
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", False)
        topology.add_feature("node_id", 0)

        topology_ext1 = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(topology_ext1)
        topology_ext1.add_feature("observed", False)
        topology_ext1.add_feature("node_id", 1)
        topology_ext1.dist = 2

        topology_ext2 = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology_ext1.add_child(topology_ext2)
        topology_ext2.add_feature("observed", False)
        topology_ext2.add_feature("node_id", 2)
        topology_ext2.dist = 1

        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents([event2], num_targets=self.num_targets)])
        topology_ext1.add_child(child2)
        child2.add_feature("observed", True)
        child2.add_feature("node_id", 3)
        child2.dist = 2

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology_ext2.add_child(child1)
        child1.add_feature("observed", True)
        child1.add_feature("node_id", 4)
        child1.dist = 1

        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event3], num_targets=self.num_targets)])
        topology_ext2.add_child(child3)
        child3.add_feature("observed", True)
        child3.add_feature("node_id", 5)
        child3.dist = 1

        branch_lens = [2, 1, 2, 1, 1]
        bifurc_model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len=0,
                branch_lens=branch_lens,
                target_lams=target_lams)
        bifurc_model.create_log_lik(
                self.exact_approximator.create_transition_matrix_wrappers(
                    bifurc_model))
        bifurc_log_lik, _ = bifurc_model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_nodes = bifurc_model._create_hazard_away_nodes([(), (TargetTract(0,0,0,0),)])
        hazard_aways = bifurc_model.sess.run(hazard_away_nodes)
        hazard_away = hazard_aways[0]
        hazard_away_from_event = hazard_aways[1]
        hazard_to_event = target_lams[0] * (1 - trim_long_probs[0]) * (1 - trim_long_probs[1])
        q_mat = np.matrix([
            [-hazard_away, hazard_to_event, hazard_away - hazard_to_event],
            [0, -hazard_away_from_event, hazard_away_from_event],
            [0, 0, 0]])
        manual_cut_log_prob = -hazard_away * tot_time
        for idx in range(2,4):
            prob_mat = tf_common._custom_expm(q_mat, branch_len_inners[idx])[0]
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
        topology.add_feature("observed", False)
        topology.add_feature("node_id", 0)

        event0 = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "atc")
        child0 = CellLineageTree(allele_events_list=[AlleleEvents([event0], num_targets=self.num_targets)])
        topology.add_child(child0)
        child0.add_feature("observed", True)
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
        child1.add_feature("observed", False)
        child1.add_feature("node_id", 2)
        child1.dist = 1

        child2 = CellLineageTree(allele_events_list=[AlleleEvents([event1], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("observed", False)
        child2.add_feature("node_id", 3)
        child2.dist = 1

        childcopy = CellLineageTree(allele_events_list = [AlleleEvents([event1], num_targets=self.num_targets)])
        child2.add_child(childcopy)
        childcopy.add_feature("observed", True)
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
        child3.add_feature("observed", True)
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
        child4.add_feature("observed", True)
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
        bifurc_model.create_log_lik(
                self.exact_approximator.create_transition_matrix_wrappers(
                    bifurc_model))
        bifurc_log_lik, _ = bifurc_model.get_log_lik()

        ####
        # Multifurcating tree calculation
        ####
        # Create collapsed tree
        # Get dist to earliest node with same ancestral state
        dist_to_earliest_ancestor = {"no_evts": {"dist_to_root": 0, "offset": 0}}
        for node in topology.traverse("preorder"):
            if node.is_root():
                continue
            if not (node.allele_events_list_str in dist_to_earliest_ancestor):
                dist_to_earliest_ancestor[node.allele_events_list_str] = {
                        "offset": node.get_distance(topology) - node.dist - dist_to_earliest_ancestor[node.up.allele_events_list_str]["dist_to_root"],
                        "br_len": node.dist,
                        "dist_to_root": node.get_distance(topology)}

        for n in topology.traverse():
            n.name = n.allele_events_list_str
            if not n.is_root():
                if n.allele_events_list_str == n.up.allele_events_list_str:
                    n.dist = 0
            n.resolved_multifurcation = False
        coll_tree = collapse_zero_lens(topology)
        coll_tree.label_node_ids()

        tot_time = 3
        branch_len_inners = [0] * 6
        for node in coll_tree.traverse():
            branch_len_inners[node.node_id] = node.dist
        branch_len_offsets = [0] * 6
        for node in coll_tree.traverse():
            if node.is_resolved_multifurcation():
                continue
            for c in node.get_children():
                branch_len_offsets[c.node_id] = dist_to_earliest_ancestor[c.allele_events_list_str]["offset"]

        multifurc_model, _  = self._create_multifurc_model(
                coll_tree,
                self.bcode_metadata,
                branch_len_inners,
                branch_len_offsets,
                target_lams = target_lams,
                tot_time = tot_time)
        multifurc_model.create_log_lik(
                self.exact_approximator.create_transition_matrix_wrappers(
                    multifurc_model))
        multifurc_log_lik, _ = multifurc_model.get_log_lik()
        self.assertTrue(np.isclose(multifurc_log_lik, bifurc_log_lik))

    def test_branch_likelihood_no_events(self):
        # Create a branch with no events
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", True)
        child = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        child.add_feature("observed", True)
        topology.add_child(child)

        branch_len = 0.1
        model = self._create_bifurc_model(topology, self.bcode_metadata, branch_len)

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Manually calculate the hazard
        hazard_away_nodes = model._create_hazard_away_nodes([()])
        hazard_aways = model.sess.run(hazard_away_nodes)
        hazard_away = hazard_aways[0]
        q_mat = np.matrix([[-hazard_away, hazard_away], [0, 0]])
        prob_mat = tf_common._custom_expm(q_mat, branch_len)[0]
        manual_log_prob = np.log(prob_mat[0,0])

        # Check the two are equal
        self.assertTrue(np.isclose(log_lik[0], manual_log_prob))

        # Calculate the hazard using approximate algo -- should be same
        t_mats = self.approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()

        self.assertTrue(np.isclose(log_lik_approx[0], manual_log_prob))

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

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
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
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", True)
        event = Event(
                start_pos = 6,
                del_len = 28,
                min_target = 0,
                max_target = 2,
                insert_str = "ATC")
        child = CellLineageTree(
                allele_events_list=[AlleleEvents([event], num_targets=self.num_targets)])
        topology.add_child(child)
        child.add_feature("observed", True)

        branch_len = 10
        model = self._create_bifurc_model(topology, self.bcode_metadata, branch_len)
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
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
        model.create_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()
        self.assertTrue(log_lik_approx < manual_log_prob)

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
        model = self._create_bifurc_model(
                topology,
                self.bcode_metadata,
                branch_len=None,
                branch_lens=[branch_len1, branch_len2])
        target_lams = model.target_lams.eval()
        trim_long_probs = model.trim_long_probs.eval()

        t_mats = self.exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
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

        # Calculate the hazard using approximate algo -- should be smaller
        t_mats = self.approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()
        self.assertTrue(log_lik_approx < manual_log_prob)

    def test_three_branch_likelihood(self):
        """
        This is just looking at how bad the approximation is.
        Not a real test
        """
        bcode_metadata = BarcodeMetadata()
        num_targets = bcode_metadata.n_targets

        # Don't self.approximate -- exact calculation
        exact_approximator = ApproximatorLB(extra_steps=10, anc_generations=5, bcode_metadata=bcode_metadata)
        # Actually do an approximation
        approximator = ApproximatorLB(extra_steps=1, anc_generations=1, bcode_metadata=bcode_metadata)

        # Create three branches: Root -- child1 -- child2 -- child3
        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("observed", True)
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
                allele_events_list=[AlleleEvents(num_targets=num_targets)])
        child2 = CellLineageTree(
                allele_events_list=[AlleleEvents(num_targets=num_targets)])
        child3 = CellLineageTree(
                allele_events_list=[AlleleEvents([event1, event2], num_targets=num_targets)])
        topology.add_child(child1)
        child1.add_child(child2)
        child2.add_child(child3)
        child1.add_feature("observed", False)
        child2.add_feature("observed", False)
        child3.add_feature("observed", True)

        branch_len = 0.5
        model = self._create_bifurc_model(topology, bcode_metadata, branch_len)

        t_mats = exact_approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        log_lik, _ = model.get_log_lik()

        # Calculate the hazard using approximate algo -- should be smaller
        t_mats = approximator.create_transition_matrix_wrappers(model)
        model.create_log_lik(t_mats)
        log_lik_approx, _ = model.get_log_lik()

        print("log lik approx", log_lik_approx)
