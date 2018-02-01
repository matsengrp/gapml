import tensorflow as tf

"""
Tensorflow helper functions
"""

def not_equal_float(a, b):
    return tf.cast(tf.not_equal(a, b), tf.float32)

def equal_float(a, b):
    return tf.cast(tf.equal(a, b), tf.float32)

def ifelse(bool_tensor, a, b):
    return bool_tensor * a + (1 - bool_tensor) * b
