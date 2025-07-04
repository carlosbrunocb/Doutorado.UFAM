import tensorflow as tf


# PSNR - Peak Signal-to-Noise Ratio
def psnr_metric(y_true, y_pred):
    """
        Calculation of Peak Signal-to-Noise Ratio (PSNR)

        Args:
          y_true: Ground truth labels
          y_pred: Predicted labels

        Returns:
          SSIM value
        """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# SSIM - Structural Similarity Index
def ssim_metric(y_true, y_pred):
    """
    Calculation of Structural Similarity Index (SSIM)

    Args:
      y_true: Ground truth labels
      y_pred: Predicted labels

    Returns:
      SSIM value
    """

    # Calculate SSIM using TensorFlow's image comparison functions
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)

    return ssim_value
