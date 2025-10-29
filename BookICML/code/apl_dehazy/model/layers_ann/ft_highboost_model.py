import tensorflow as tf
from tensorflow.keras import layers


# High-Boost filter layer
# G = I + A * Mask
# Mask = I - I_blurred
class HighBoostLayer(layers.Layer):
    """
    A Keras layer that applies the High-Boost filter to enhance edges.

    This layer expects four inputs:
        - v: The V (luminance) channel extracted from the original HSV image.
             Expected shape: [batch_size, height, width, 1].
        - v_blurred: The V channel of the HSV image, pre-filtered with a
                     low-pass filter (e.g., Gaussian blur). This is typically
                     the 'learned' blurry version.
                     Expected shape: [batch_size, height, width, 1].
        - rgb_image: The original RGB image. This is used to extract H and S
                     channels for reconstruction.
                     Expected shape: [batch_size, height, width, 3].
        - a: The boost factor (amplification factor) for the mask enhancement.
             This is typically a 'learned' parameter.
             Expected shape: [batch_size, 1, 1, 1] or broadcastable variations like [batch_size, 1].

    The 'v_blurred' and the 'a' factor are typically outputs from other parts of a neural network,
    allowing the network to learn the blurring process and the amplification strength.
    """

    def __init__(self, **kwargs):
        super(HighBoostLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        :param inputs: [v, v_blurred, rgb_image, a],
            - v: V channel of HSV image (float32 tensor, shape [B, H, W, 1])
            - v_blurred: blurred V channel (float32 tensor, shape [B, H, W, 1])
            - image_rgb: original RGB image (float32 tensor, shape [B, H, W, 3])
            - a: amplification factor (float32 tensor, shape [B, 1, 1, 1] or [B, 1])
        :return: sharpened_rgb: filtered image (float32 tensor, shape [B, H, W, 3])
        """
        if not isinstance(inputs, list) or len(inputs) != 4:
            raise ValueError("The HighBoostLayer layer expects a list of 4 tensors: "
                             "[V, V_blurred, rgb_image, A]. Got {} inputs.".format(len(inputs)))

        v, v_blurred, image_rgb, a = inputs

        # check tensor shapes
        tf.debugging.assert_shapes([
            (v, [None, None, None, 1]),
            (v_blurred, [None, None, None, 1]),
            (image_rgb, [None, None, None, 3]),
            (a, [None, 1])
        ], message="Shape mismatch nos inputs da HighBoostLayer")

        # expanding to a = [B, 1, 1, 1] for broadcasting
        a = tf.reshape(a, (-1, 1, 1, 1))

        # convert RGB to HSV
        image_hsv = tf.image.rgb_to_hsv(image_rgb)
        h, s, _ = tf.split(image_hsv, num_or_size_splits=3, axis=-1)

        # applying High-Boost filter
        mask = v - v_blurred
        v_sharpened = v + a * mask

        # clamp the values to the output range [0, 1]
        v_sharpened = tf.clip_by_value(v_sharpened, 0.0, 1.0)

        # Reconstruct HSV image with modified V channel
        sharpened_hsv = tf.concat([h, s, v_sharpened], axis=-1)

        # Convert back to RGB
        sharpened_rgb = tf.image.hsv_to_rgb(sharpened_hsv)

        return sharpened_rgb
