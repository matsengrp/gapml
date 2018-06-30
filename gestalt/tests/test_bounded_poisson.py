import unittest

import numpy as np
from bounded_poisson import PaddedBoundedPoisson, ZeroInflatedBoundedPoisson

class BoundedPoissonTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_zero_inflated(self):
        min_val = 0
        max_val = 3
        bounded_poiss = ZeroInflatedBoundedPoisson(min_val, max_val, 1)
        size = 1000000
        rvs = bounded_poiss.rvs(size)
        self.assertTrue(np.all(np.unique(rvs) == np.arange(min_val, max_val + 1)))
        for i in range(min_val, max_val + 1):
            self.assertTrue(np.abs(np.sum(rvs == i)/float(size) - bounded_poiss.pmf(i)) < 0.001)

        min_val = 3
        max_val = 6
        bounded_poiss = ZeroInflatedBoundedPoisson(min_val, max_val, 1)
        rvs = bounded_poiss.rvs(size)
        self.assertTrue(np.all(np.unique(rvs) == np.arange(min_val, max_val + 1)))
        for i in range(min_val, max_val + 1):
            self.assertTrue(np.abs(np.sum(rvs == i)/float(size) - bounded_poiss.pmf(i)) < 0.001)

    def test_padded(self):
        size = 1000000
        min_val = 3
        max_val = 6
        bounded_poiss = PaddedBoundedPoisson(min_val, max_val, 1)
        self.assertTrue(np.isclose(
            np.sum([bounded_poiss.pmf(i) for i in range(min_val, max_val + 1)]), 1.0))
        rvs = bounded_poiss.rvs(size)
        self.assertTrue(np.all(np.unique(rvs) == np.arange(min_val, max_val + 1)))
        for i in range(min_val, max_val + 1):
            self.assertTrue(np.abs(np.sum(rvs == i)/float(size) - bounded_poiss.pmf(i)) < 0.001)

        min_val = 0
        max_val = 3
        bounded_poiss = PaddedBoundedPoisson(min_val, max_val, 1)
        self.assertTrue(np.isclose(
            np.sum([bounded_poiss.pmf(i) for i in range(min_val, max_val + 1)]), 1.0))
        rvs = bounded_poiss.rvs(size)
        self.assertTrue(np.all(np.unique(rvs) == np.arange(min_val, max_val + 1)))
        for i in range(min_val, max_val + 1):
            self.assertTrue(np.abs(np.sum(rvs == i)/float(size) - bounded_poiss.pmf(i)) < 0.001)
