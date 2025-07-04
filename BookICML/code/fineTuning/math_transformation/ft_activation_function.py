import tensorflow as tf


def shifted_relu(x):
    return tf.nn.relu(x) + 1e-4
