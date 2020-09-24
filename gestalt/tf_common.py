import time
import tensorflow as tf
import numpy as np
import scipy.linalg

from common import get_randint

"""
Tensorflow helper functions
"""


def scatter_nd(
        index_vals,
        output_shape,
        default_value=0,
        name=None):
    """
    A custom version of scatter_nd

    @param index_vals: List of [[index0, index1, ..., index_last] val], unordered
    @param output_shape: same as scatter_nd
    @param name: name of tensor
    """
    sparse_indices = [tup[0] for tup in index_vals]
    sparse_vals = [tup[1] - default_value for tup in index_vals]

    new_sparse_mat = tf.scatter_nd(
        sparse_indices,
        sparse_vals,
        output_shape) + default_value
    return new_sparse_mat

def not_equal_float(a, b):
    return tf.cast(tf.not_equal(a, b), tf.float64)

def equal_float(a, b):
    return tf.cast(tf.equal(a, b), tf.float64)

def greater_equal_float(a, b):
    return tf.cast(tf.greater_equal(a, b), tf.float64)

def less_float(a, b):
    return tf.cast(tf.less(a, b), tf.float64)

def ifelse(bool_tensor, a, b):
    return bool_tensor * a + (1 - bool_tensor) * b

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    Define custom py_func which takes also a grad op as argument:
    (Code was copied from the internet)
    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(int(np.abs(get_randint() + int(time.time()) - 1000000000))) + str(get_randint())

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def _custom_expm(x, t):
    """
    My own expm implementation
    The additional outputs are useful for calculating the gradient later
    """
    D, A = np.linalg.eig(x)
    #print("D", D.shape, np.unique(D).size)
    A_inv = np.linalg.inv(A)
    if np.sum(np.isnan(D)) + np.sum(np.isnan(A_inv)) + np.sum(np.isnan(A)) > 0:
        print("A_INV", A_inv)
        print("bad A", A)
        print("bad D", D)
        1/0
    #res =  np.dot(A, np.dot(
    #    np.diag(np.exp(
    #        # Do not allow overflow!
    #        np.minimum(D * t, 10))),
    #    A_inv))
    res = scipy.linalg.expm(x * t)
    return [res, A, A_inv, D]

def _expm_grad(op, grad0, grad1, grad2, grad3):
    """
    @param op: The `expm` `Operation` that we are differentiating, which we can use
                to find the inputs and outputs of the original op.
    @param grad[0,1,2,3]: Gradient with respect to each of the outputs of the `expm` op.

    @return the gradient with respect to each input
            (This will be assuming that only the first output of expm is used
            and the rest is not used to calculate the final value.)


    Gradient calculations based on Kalbfleisch (1985)
    """
    A = op.outputs[1]
    A_inv = op.outputs[2]
    D = op.outputs[3]
    Q = op.inputs[0]
    t = op.inputs[1]
    Q_len = Q.shape[0]

    D_vec = tf.reshape(D, (Q_len, 1), name="D_vec")
    expDt = tf.exp(D * t, name="expDt")
    expDt_vec = tf.reshape(expDt, (Q_len, 1))
    DD_diff = tf.subtract(D_vec, tf.transpose(D_vec), name="DD_diff_raw")
    # Make sure that the difference matrix doesnt have things exactly equal to zero
    # Perturb a little bit if that happens.
    bad_diff_thres = 1e-4
    bad_diffs = tf.cast(tf.less(tf.abs(DD_diff), bad_diff_thres), dtype=tf.float64)
    DD_diff = tf.add(
            DD_diff,
            tf.multiply(bad_diffs, bad_diff_thres))
    DD_diff = tf.matrix_set_diag(DD_diff, tf.ones(Q_len, dtype=tf.float64), name="DD_diff_set_diag")
    t_factor = tf.divide(expDt_vec - tf.transpose(expDt_vec), DD_diff, name="t_factor_raw")
    t_factor = tf.matrix_set_diag(t_factor, t * expDt, name="t_factor_filled")

    A_inv_col = tf.reshape(tf.transpose(A_inv), (-1, 1))
    A_row = tf.reshape(A, (1, -1))
    Gs = tf.matmul(A_inv_col, A_row)
    Ts = tf.tile(t_factor, multiples=[Q_len, Q_len])
    Vs = tf.multiply(Gs, Ts)

    # Derivation
    # dL_dQij = dotproduct(dL/dP, dP/dQij)
    #         = trace(dL/dP dP/dQij.T)
    #         = trace(dL/dP Ainv.T Vij.T A.T)
    #         = trace(A.T dL/dP Ainv.T Vij.T)
    #         = dotproduct(A.T dL/dP Ainv.T, Vij)
    sandwicher = tf.matmul(
            A,
            tf.matmul(grad0, A_inv, transpose_b=True),
            transpose_a=True)
    tiled_sandwicher = tf.tile(sandwicher, multiples=[Q_len, Q_len])
    sw_Vs = tf.multiply(tiled_sandwicher, Vs)
    sw_Vs = tf.verify_tensor_all_finite(sw_Vs, "sw vs before cast")
    # Scaling for numerical overflow issues
    mean_sw_Vs = tf.reduce_mean(sw_Vs)
    scaled_sw_Vs = sw_Vs/mean_sw_Vs
    scaled_sw_Vs = tf.cast(scaled_sw_Vs, tf.float32)
    scaled_sw_Vs = tf.verify_tensor_all_finite(scaled_sw_Vs, "sw vs after cast")
    scaled_sw_Vs = tf.expand_dims(tf.expand_dims(scaled_sw_Vs, 0), -1)

    # Calculate the dotproduct using convolutional operators
    # TODO: unfortunately the conv operator in tensorflow (v1.5) requires float32 instead of 64.
    #       We're going to lose some precision in the tradeoff for using someone else's code.
    #       In the future, we can consider implementing this entirely on our own.
    conv_filter = tf.ones(shape=[Q_len, Q_len], dtype=tf.float32)
    reshaped_conv_filter = tf.expand_dims(tf.expand_dims(conv_filter, -1), -1)
    avged = tf.nn.conv2d(scaled_sw_Vs, reshaped_conv_filter, strides=[1, Q_len, Q_len, 1], padding="SAME")
    # Fixing up after we rescaled
    dL_dQ = tf.cast(avged[0, :, :, 0], dtype=tf.float64) * mean_sw_Vs

    dP_dt = tf.matmul(A, tf.matmul(
                tf.diag(D * expDt),
                A_inv))
    dL_dt = tf.reduce_sum(tf.multiply(grad0, dP_dt))

    return tf.cast(dL_dQ, tf.float64), dL_dt

def myexpm(Q, t, name=None):
    """
    @param Q: the instantaneous transition matrix
    @param t: the time

    @return tensorflow object with exp(Qt)
    """
    with tf.name_scope(name, "Myexpm", [Q, t]) as name:
        expm_wrapped_func = py_func(_custom_expm,
                        [Q, t],
                        [tf.float64, tf.float64, tf.float64, tf.float64],
                        name=name,
                        grad=_expm_grad)
        return expm_wrapped_func
