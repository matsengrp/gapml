import unittest

import numpy as np

from allele_events import AlleleEvents
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_estimator import CLTLassoEstimator
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from approximator import ApproximatorLB

class LikelihoodCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_metadata = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TT"),
            cut_sites = [3, 3],
            crucial_pos_len = [3,3])
        self.num_targets = self.bcode_metadata.n_targets

    def test_branch_likelihood(self):
        # Don't approximate -- exact calculation
        approximator = ApproximatorLB(extra_steps=10, anc_generations=3, bcode_metadata=self.bcode_metadata)

        # Create a branch with no events
        topology = CellLineageTree()
        child = CellLineageTree(allele_events=AlleleEvents([], num_targets=self.num_targets))
        topology.add_child(child)

        model = CLTLikelihoodModel(topology, self.bcode_metadata)
        branch_lens = [0, 0]
        branch_lens[child.node_id] = 10
        model.set_vals(
            branch_lens,
            target_lams = np.ones(self.num_targets),
            trim_long_probs = 0.1 * np.ones(2),
            trim_zero_prob = 0.5,
            trim_poisson_params = np.ones(2),
            insert_zero_prob = 0.5,
            insert_poisson_param = 2,
            cell_type_lams = None)

        lik_calculator = CLTLassoEstimator(0, model, approximator)

        log_lik = lik_calculator.get_likelihood(model)
        print(log_lik)
