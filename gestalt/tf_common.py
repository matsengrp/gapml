import tensorflow as tf
import numpy as np
import scipy.linalg

import numpy as np

"""
Tensorflow helper functions
"""

def scatter_nd(
        index_vals,
        output_shape,
        default_value = 0,
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

def ifelse(bool_tensor, a, b):
    return bool_tensor * a + (1 - bool_tensor) * b

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    Define custom py_func which takes also a grad op as argument:
    (Code was copied from the internet)
    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

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
    A_inv = np.linalg.inv(A)
    res =  np.dot(A, np.dot(
        np.diag(np.exp(D * t)),
        A_inv))
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
    TODO: implement a faster version
    TODO: do some checks on this gradient
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
    DD_diff = tf.matrix_set_diag(DD_diff, tf.ones(Q_len, dtype=tf.float64), name="DD_diff_set_diag")
    t_factor = tf.divide(expDt_vec - tf.transpose(expDt_vec), DD_diff, name="t_factor_raw")
    t_factor = tf.matrix_set_diag(t_factor, t * expDt, name="t_factor_filled")
    dL_dQ = []
    for i in range(Q_len):
        dL_dQi = []
        for j in range(Q_len):
            A_invi = tf.reshape(A_inv[:,i], (Q_len, 1))
            A_j = tf.reshape(A[j,:], (1, Q_len))
            G = tf.matmul(A_invi, A_j)
            V = tf.multiply(G, t_factor)
            dP_dQij = tf.matmul(A, tf.matmul(V, A_inv))
            dL_dQi.append(
                    tf.reduce_sum(tf.multiply(grad0, dP_dQij)))

        dL_dQ.append(tf.parallel_stack(dL_dQi))

    dL_dQ = tf.parallel_stack(dL_dQ)

    dP_dt = tf.matmul(A, tf.matmul(
                tf.diag(D * expDt),
                A_inv))
    dL_dt = tf.reduce_sum(tf.multiply(grad0, dP_dt))

    return dL_dQ, dL_dt

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
