import tensorflow as tf
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

