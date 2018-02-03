import tensorflow as tf
import numpy as np
import scipy.linalg

import numpy as np

"""
Tensorflow helper functions
"""

def not_equal_float(a, b):
    return tf.cast(tf.not_equal(a, b), tf.float32)

def equal_float(a, b):
    return tf.cast(tf.equal(a, b), tf.float32)

def ifelse(bool_tensor, a, b):
    return bool_tensor * a + (1 - bool_tensor) * b

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def custom_expm(x, t):
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
    @param grad: Gradient with respect to the output of the `expm` op.
    """
    #for i in range(4):
    #    print(op.outputs[i])
    #print(grad0)
    #print(grad1)
    #print(grad2)
    #print(grad3)
    # A bogus gradient to see what happens
    return op.outputs[1] * grad0 * op.outputs[2], tf.constant(3)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def myexpm(Q, t, name=None):
    """
    @param Q: the instantaneous transition matrix
    @param t: the time

    @return tensorflow object with exp(Qt)
    """
    with tf.name_scope(name, "Myexpm", [Q, t]) as name:
        expm_wrapped_func = py_func(custom_expm,
                        [Q, t],
                        [tf.float32, tf.float32, tf.float32, tf.float32],
                        name=name,
                        grad=_expm_grad)
        return expm_wrapped_func
