import tensorflow as tf


def shifted_relu(x):
    return tf.nn.relu(x) + 1e-4


def linear_plus_eps(x):
    return x + 1e-6
