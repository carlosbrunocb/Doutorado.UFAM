import tensorflow as tf
from tensorflow.keras.layers import Layer


class MinMaxNormalize(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """
        Keras layer to perform per-sample Min-Max normalization.
        Normalizes input tensor values to the range [0, 1].

        Args:
            epsilon (float): A small value added to the denominator to prevent
                             division by zero during normalization.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(MinMaxNormalize, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        """
        Applies Min-Max normalization to the input tensor.

        Args:
            inputs (tf.Tensor): The input tensor to be normalized. Expected shape
                                is typically (batch, height, width, channels).

        Returns:
            tf.Tensor: The normalized tensor, with values between 0 and 1.
        """
        # Calculate min and max values per sample (batch item) across spatial and channel dimensions
        min_val = tf.reduce_min(inputs, axis=[1, 2, 3], keepdims=True)
        max_val = tf.reduce_max(inputs, axis=[1, 2, 3], keepdims=True)

        return (inputs - min_val) / (max_val - min_val + self.epsilon)  # avoid division by zero

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape
        return input_shape

    def get_config(self):
        """
        Returns the configuration of the layer. Required for serialization.
        """
        config = super(MinMaxNormalize, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


class PowerLawTransformLayer(Layer):
    def __init__(self, clip_output=True, output_range=(0.0, 1.0), clamping_epsilon=1e-6, **kwargs):
        """
        Keras layer to apply a power transformation with offset:
        f(x) = alpha * (x + epsilon) ^ gamma.
            alpha (float): The constant 'alpha' in the formula.
            epsilon (float): The constant 'epsilon' in the formula. Used to avoid problems
                  with log/pow of zero and to add saturation gain to pixel intensities.
            gamma (float): The constant 'gamma' in the formula (the exponent).

        Args:
            clip_output (bool): If True, output values will be clipped to the 'output_range'.
                      Recommended to keep valid pixel values within the dynamic range.
                      Default = 2^8 (8 bits)
            output_range (tuple): A tuple (min_val, max_val) to adjust the output values.
                      Default (0.0, 1.0).
            clamping_epsilon: Small value to avoid division by zero.
            **kwargs: Additional arguments for the Layer base class.
        """
        super(PowerLawTransformLayer, self).__init__(**kwargs)
        self.clip_output = clip_output
        self.output_range = tuple(tf.cast(v, tf.float32) for v in output_range)
        self.clamping_epsilon = clamping_epsilon

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
        clamped_plus_eps = tf.clip_by_value(image_tensor + epsilon, self.clamping_epsilon, tf.float32.max)
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
            transformed = (transformed - min_vals) / (max_vals - min_vals + self.clamping_epsilon)

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
            "output_range": tuple(float(v) for v in self.output_range),
            "clamping_epsilon": self.clamping_epsilon
        })
        return config


class PowerLawTransformWithDepthMapMasksLayer(Layer):
    def __init__(self, clip_output=True, output_range=(0.0, 1.0), clamping_epsilon=1e-6, **kwargs):
        """
        Keras layer to apply a power transformation with offset:
            f(x) = alpha * (x + epsilon) ^ gamma.
              var alpha (float): The constant 'alpha' in the formula.
              var epsilon (float): The constant 'epsilon' in the formula. Used to avoid problems
                      with log/pow of zero and to add saturation gain to pixel intensities.
              var gamma (float): The constant 'gamma' in the formula (the exponent).

        And use depth map masks to divide the images into local regions to apply the transformation.

        Args:
            clip_output (bool): If True, output values will be clipped to the 'output_range'.
                      Recommended to keep valid pixel values within the dynamic range.
                      Default = 2^8 (8 bits)
            output_range (tuple): A tuple (min_val, max_val) to adjust the output values.
                      Default (0.0, 1.0).
            clamping_epsilon (float): Small value used to avoid issues with log/pow of zero
                      or division by zero during normalization.
            **kwargs: Additional arguments for the Layer base class.
        """
        super(PowerLawTransformWithDepthMapMasksLayer, self).__init__(**kwargs)
        self.clip_output = clip_output
        self.output_range = tuple(tf.cast(v, tf.float32) for v in output_range)
        self.clamping_epsilon = clamping_epsilon

    def call(self, inputs):
        # inputs is list of tensors: [image_tensor, predicted_params_tensor, batch_mask]
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
        clamped = tf.clip_by_value(image_exp + epsilon, self.clamping_epsilon, tf.float32.max)
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
            final_image = (final_image - min_vals) / (max_vals - min_vals + self.clamping_epsilon)

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
            "output_range": tuple(float(v) for v in self.output_range),
            "clamping_epsilon": self.clamping_epsilon
        })
        return config


# Compute the dehazy function:
#   I(x) = (H(x) - A * (1.0 - t(x))) / t(x)
#   t(x) = e^(-B * d(x))
#   where:
#     * I(x) = dehazy image
#     * H(x) = hazy image
#     * t(x) = medium transmission map
#     * d(x) = depth map
#     * A    = atmospheric light
#     * B    = atmospheric dispersion coefficient
class DehazeLayer(tf.keras.layers.Layer):
    """
    A Keras layer that applies the physical dehaze model to an image.

    I(x) = (H(x) - A * (1.0 - t(x))) / t(x)
    t(x) = e^(-B * d(x))

    Inputs:
        hazy (tf.Tensor): The hazy RGB image.
        depth_map (tf.Tensor): The depth map.
        b (tf.Tensor): The atmospheric dispersion coefficient.
        a (tf.Tensor): The global atmospheric light.

    Output:
        tf.Tensor: The dehazed RGB image, clipped to [0, 1].
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super(DehazeLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        """
        inputs:
            hazy (tf.Tensor): hazy image, shape (batch, H, W, 3)
            depth_map (tf.Tensor): depth map, shape (batch, H, W, 1)
            predicted_params (tf.Tensor): shape (batch, 2)
                b is atmospheric dispersion coefficient
                a is atmospheric light
        """
        hazy, depth_map, predicted_params = inputs

        # Extracts the predicted parameters for each image in the batch
        # a, b
        a = predicted_params[:, 0:1]
        b = predicted_params[:, 1:2]

        # Ensure all inputs are float32 (or other desired dtype)
        hazy = tf.cast(hazy, self.dtype)  # self.dtype is inherited from Layer
        depth_map = tf.cast(depth_map, self.dtype)
        b = tf.cast(b, self.dtype)
        a = tf.cast(a, self.dtype)

        # Ensure 'a' has shape (batch, 1)
        a = tf.reshape(a, (tf.shape(a)[0], 1))

        # Step 1: compute t(x)
        b = tf.reshape(b, (tf.shape(b)[0], 1, 1, 1))  # shape (batch, 1, 1, 1)
        t = tf.exp(-b * depth_map)  # shape (batch, H, W, 1)

        # Step 2: Expand t(x) to 3 channels
        t_rgb = tf.broadcast_to(t, tf.shape(hazy))  # shape (batch, H, W, 3)

        # Step 3: Expand 'a' to (batch, 1, 1, 3)
        a_rgb = tf.reshape(a, (tf.shape(a)[0], 1, 1, 1))
        a_rgb = tf.broadcast_to(a_rgb, tf.shape(hazy))  # shape (batch, H, W, 3)

        # Step 4: Apply the dehaze equation
        dehazy = (hazy - a_rgb * (1.0 - t_rgb)) / (t_rgb + self.epsilon)  # avoid division by zero

        # Clip to keep values between 0 and 1
        dehazy = tf.clip_by_value(dehazy, 0.0, 1.0)

        return dehazy

    def compute_output_shape(self, input_shape):
        # input_shape : [shape_hazy, shape_depth_map, shape_b, shape_a]
        # Checks if the number of entries
        if len(input_shape) != 3:
            raise ValueError('DehazeLayer expects 3 input tensors.')

        # The output shape is the same as the hazy image (first input)
        return input_shape[0]

    def get_config(self):
        config = super(DehazeLayer, self).get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config
