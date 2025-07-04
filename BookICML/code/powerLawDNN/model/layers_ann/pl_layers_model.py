import tensorflow as tf
from tensorflow.keras.layers import Layer


class MinMaxNormalize(Layer):
    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=[1, 2, 3], keepdims=True)
        max_val = tf.reduce_max(inputs, axis=[1, 2, 3], keepdims=True)

        return (inputs - min_val) / (max_val - min_val + 1e-8)  # evita divisão por zero


class PowerLawTransformLayer(Layer):
    def __init__(self, clip_output=True, output_range=(0.0, 1.0), **kwargs):
        """
        Keras layer to apply a power transformation with offset:
        f(x) = alpha * (x + epsilon) ^ gamma.
          :param alpha (float): The constant 'alpha' in the formula.
          :param epsilon (float): The constant 'epsilon' in the formula. Used to avoid problems
                  with log/pow of zero and to add saturation gain to pixel intensities.
          :param gamma (float): The constant 'gamma' in the formula (the exponent).

        Args:
            :param clip_output (bool): If True, output values will be clipped to the 'output_range'.
                      Recommended to keep valid pixel values within the dynamic range.
                      Default = 2^8 (8 bits)
            :param output_range (tuple): A tuple (min_val, max_val) to adjust the output values.
                      Default (0.0, 1.0).
            :param **kwargs: Additional arguments for the Layer base class.
        """
        super(PowerLawTransformLayer, self).__init__(**kwargs)
        self.clip_output = clip_output
        self.output_range = tuple(tf.cast(v, tf.float32) for v in output_range)

    def call(self, inputs):
        # inputs is list of tensors: [image_tensor, predicted_params_tensor]
        image_tensor = tf.cast(inputs[0], tf.float32)  # shape (batch_size, height, width, channels)
        predicted_params_tensor = inputs[1]  # shape (batch_size, 3)

        # Extracts the predicted parameters for each image in the batch
        # Reshape the parameters to be compatible with the broadcast image
        # O [:, X:X+1]
        # (batch_size, 1) -> (batch_size, 1, 1, 1)
        alpha = tf.reshape(predicted_params_tensor[:, 0:1], (-1, 1, 1, 1))
        epsilon = tf.reshape(predicted_params_tensor[:, 1:2], (-1, 1, 1, 1))
        gamma = tf.reshape(predicted_params_tensor[:, 2:3], (-1, 1, 1, 1))

        # f(x) = alpha * (x + epsilon)^gamma
        # Step 1: (x + epsilon) ^ gamma
        # Handle cases where temp_result can be negative and gamma is fractional to avoid NaN.
        # We clip temp_result to ensure it is non-negative before raising it to the power.
        # Use 1e-8 to avoid log(0) and 0^0 or 0^neg
        clamped_plus_eps = tf.clip_by_value(image_tensor + epsilon, 1e-8, tf.float32.max)
        power_result = tf.pow(clamped_plus_eps, gamma)

        # Step 2: alpha * [(x + epsilon) ^ gamma]
        transformed = alpha * power_result

        # Optional: Clamp the values to the desired output range
        if self.clip_output:
            transformed = tf.clip_by_value(transformed, self.output_range[0], self.output_range[1])
        else:
            # Per-sample min-max normalization
            min_vals = tf.reduce_min(transformed, axis=[1, 2, 3], keepdims=True)
            max_vals = tf.reduce_max(transformed, axis=[1, 2, 3], keepdims=True)
            transformed = (transformed - min_vals) / (max_vals - min_vals + 1e-8)

        return transformed

    def compute_output_shape(self, input_shape):
        # input_shape is list of tensors: [image_shape, params_shape]
        return input_shape[0]  # The output has the same shape as the input image

    def get_config(self):
        # Method is required so that the layer can be serialized and deserialized
        # (saved and loaded with the model)
        config = super(PowerLawTransformLayer, self).get_config()
        config.update({
            "clip_output": self.clip_output,
            "output_range": tuple(float(v) for v in self.output_range)
        })
        return config


class PowerLawTransformWithDepthMapMasksLayer(Layer):
    def __init__(self, clip_output=True, output_range=(0.0, 1.0), **kwargs):
        """
        Keras layer to apply a power transformation with offset:
            f(x) = alpha * (x + epsilon) ^ gamma.
              var alpha (float): The constant 'alpha' in the formula.
              var epsilon (float): The constant 'epsilon' in the formula. Used to avoid problems
                      with log/pow of zero and to add saturation gain to pixel intensities.
              var gamma (float): The constant 'gamma' in the formula (the exponent).

        And use depth map masks to divide the images into local regions to apply the transformation.

        Args:
            :param clip_output (bool): If True, output values will be clipped to the 'output_range'.
                      Recommended to keep valid pixel values within the dynamic range.
                      Default = 2^8 (8 bits)
            :param output_range (tuple): A tuple (min_val, max_val) to adjust the output values.
                      Default (0.0, 1.0).
            :param **kwargs: Additional arguments for the Layer base class.
        """
        super(PowerLawTransformWithDepthMapMasksLayer, self).__init__(**kwargs)
        self.clip_output = clip_output
        self.output_range = tuple(tf.cast(v, tf.float32) for v in output_range)

    def call(self, inputs):
        # inputs is list of tensors: [image_tensor, predicted_params_tensor]
        # B batch size, H height, W width, C channels, M number of masks
        image_tensor = tf.cast(inputs[0], tf.float32)  # (B, H, W, C)
        predicted_params_tensor = inputs[1]  # (B, 3*M)
        batch_mask = tf.cast(inputs[2], tf.float32)  # (B, M, H, W)

        B, H, W, C = tf.unstack(tf.shape(image_tensor))
        M = tf.shape(batch_mask)[1]

        # Extracts the predicted parameters for each image in the batch
        # Alpha, Epsilon, Gamma: each (B, M)
        alpha = predicted_params_tensor[:, 0::3]  # (B, M)
        epsilon = predicted_params_tensor[:, 1::3]
        gamma = predicted_params_tensor[:, 2::3]

        # Reshape the parameters to be compatible with the broadcast:
        # (B, M, 1, 1, 1)
        alpha = tf.reshape(alpha, (B, M, 1, 1, 1))
        epsilon = tf.reshape(epsilon, (B, M, 1, 1, 1))
        gamma = tf.reshape(gamma, (B, M, 1, 1, 1))

        # Expand image to (B, M, H, W, C)
        image_exp = tf.expand_dims(image_tensor, 1)  # (B, 1, H, W, C)
        image_exp = tf.tile(image_exp, [1, M, 1, 1, 1])  # (B, M, H, W, C)

        # f(x) = alpha * (x + epsilon)^gamma
        # Step 1: (x + epsilon) ^ gamma
        # Handle cases where temp_result can be negative and gamma is fractional to avoid NaN.
        # We clip temp_result to ensure it is non-negative before raising it to the power.
        # Use 1e-8 to avoid log(0) and 0^0 or 0^neg
        clamped = tf.clip_by_value(image_exp + epsilon, 1e-8, tf.float32.max)
        power_result = tf.pow(clamped, gamma)
        # Step 2: alpha * [(x + epsilon) ^ gamma]
        transformed = alpha * power_result  # (B, M, H, W, C)

        # Apply masks: batch_mask -> (B, M, H, W, 1)
        mask_exp = tf.expand_dims(batch_mask, -1)
        masked_transformed = transformed * mask_exp  # (B, M, H, W, C)

        # Sum over M regions to generate final image (B, H, W, C)
        final_image = tf.reduce_sum(masked_transformed, axis=1)

        # Optionfinal_imageal: Clamp the values to the desired output range
        if self.clip_output:
            final_image = tf.clip_by_value(final_image, self.output_range[0], self.output_range[1])
        else:
            # Per-sample min-max normalization
            min_vals = tf.reduce_min(final_image, axis=[1, 2, 3], keepdims=True)
            max_vals = tf.reduce_max(final_image, axis=[1, 2, 3], keepdims=True)
            final_image = (final_image - min_vals) / (max_vals - min_vals + 1e-8)

        return final_image

    def compute_output_shape(self, input_shape):
        # input_shape is list of tensors: [image_shape, params_shape]
        return input_shape[0]  # The output has the same shape as the input image

    def get_config(self):
        # Method is required so that the layer can be serialized and deserialized
        # (saved and loaded with the model)
        config = super(PowerLawTransformWithDepthMapMasksLayer, self).get_config()
        config.update({
            "clip_output": self.clip_output,
            "output_range": tuple(float(v) for v in self.output_range)
        })
        return config
