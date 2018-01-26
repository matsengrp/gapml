import unittest

import numpy as np
from transition_matrix import TransitionMatrixWrapper

class TransitionMatrixTestCase(unittest.TestCase):
    def testEqual(self):
        # Check decomposition is correct
        matrix_dict = {
            "0": {"0": 1, "1": 0.1},
            "1": {"1":2}}
        mat = TransitionMatrixWrapper(matrix_dict)
        self.assertTrue(
            np.all(mat.matrix == np.dot(mat.A, np.dot(np.diag(mat.D), mat.A_inv))))
