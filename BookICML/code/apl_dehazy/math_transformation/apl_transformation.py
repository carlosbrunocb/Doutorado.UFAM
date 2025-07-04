import tensorflow as tf


# Normalization between 0 and 1
def normalize_to_0_1(images):
    min_vals = tf.reduce_min(images, axis=[1, 2, 3], keepdims=True)
    max_vals = tf.reduce_max(images, axis=[1, 2, 3], keepdims=True)
    normalized = (images - min_vals) / (max_vals - min_vals + 1e-8)  # avoid division by zero
    return normalized
