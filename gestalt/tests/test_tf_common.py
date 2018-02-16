import unittest

import numpy as np
import tensorflow as tf

import tf_common
from scipy.optimize import check_grad

class LikelihoodCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.g_opt = tf.train.GradientDescentOptimizer(1)
        self.sess = tf.InteractiveSession()

    def test_expm_grad(self):
        Q_orig_val = np.array([[5.0, 2.3],[4,6]])
        t_orig_val = 0.1
        Q = tf.Variable(Q_orig_val, dtype=tf.float64)
        t = tf.Variable(t_orig_val, dtype=tf.float64)

        Q_ph = tf.placeholder(tf.float64, shape=[2,2])
        Q_assign = Q.assign(Q_ph)

        t_ph = tf.placeholder(tf.float64, shape=[])
        t_assign = t.assign(t_ph)

        p_mat, _, _, _ = tf_common.myexpm(Q, t)
        p_mat_sum = tf.reduce_sum(p_mat)
        p_mat_sum_grads = self.g_opt.compute_gradients(p_mat_sum, var_list=[Q,t])

        tf.global_variables_initializer().run()

        eps = 1e-6
        my_sum, my_grads = self.sess.run([p_mat_sum, p_mat_sum_grads])
        Q_grad = my_grads[0][0]
        t_grad = my_grads[1][0]
        print("QQQ", Q_grad)
        # Check gradient with respect to Q
        for i in range(2):
            for j in range(2):
                Q_new_val = np.copy(Q_orig_val)
                Q_new_val[i,j] += eps
                self.sess.run(Q_assign, feed_dict={Q_ph: Q_new_val})
                my_sum_eps = self.sess.run(p_mat_sum)
                approx_grad = (my_sum_eps - my_sum)/eps
                print("approx grad", approx_grad)
                print("qqq", Q_grad[i,j])
                self.assertTrue(np.isclose(approx_grad,  Q_grad[i,j]))

        # Check gradient with respect to t
        self.sess.run(Q_assign, feed_dict={Q_ph: Q_orig_val})
        self.sess.run(t_assign, feed_dict={t_ph: t_orig_val + eps})
        my_sum_eps = self.sess.run(p_mat_sum)
        approx_grad = (my_sum_eps - my_sum)/eps
        self.assertTrue(np.isclose(approx_grad,  t_grad))
