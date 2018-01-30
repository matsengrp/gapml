import unittest

import numpy as np
import scipy.linalg
import tensorflow as tf

from indel_sets import TargetTract
from clt_likelihood_model import CLTLikelihoodModel
from clt_likelihood_model_tensorflow import CLTLikelihoodModelCalculator
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata

class CLTLiklihooodTensorflowTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_metadata = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_sites = [3, 3, 3],
            crucial_pos_len = [3,3])

    def test_hazard(self):
        topology = CellLineageTree()
        model = CLTLikelihoodModel(topology, self.bcode_metadata)
        calculator = CLTLikelihoodModelCalculator(model)

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            hazard, hazard_grad = sess.run([calculator.hazard, calculator.hazard_grad], feed_dict={
                calculator.hazard_ph: [0, 0],
                calculator.hazard_long_status_ph: [0, 0]})
            print(hazard)
            a = model.get_hazard(TargetTract(0,0,0,0))
            print("a", a)
            print(hazard_grad, "GRAD")

    def test_hazard_away(self):
        topology = CellLineageTree()
        model = CLTLikelihoodModel(topology, self.bcode_metadata)
        calculator = CLTLikelihoodModelCalculator(model)

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()

            tts = (TargetTract(0,0,0,0),)
            left_m, right_m = model._get_hazard_masks(tts)
            hazard_away, hazard_away_grad = sess.run(
                [calculator.hazard_away, calculator.hazard_away_grad],
                feed_dict={
                    calculator.left_trimmables_ph: left_m,
                    calculator.right_trimmables_ph: right_m})
            print(hazard_away)
            print(model.get_hazard_away(tts))
            print(hazard_away_grad, "GRADDDD")
