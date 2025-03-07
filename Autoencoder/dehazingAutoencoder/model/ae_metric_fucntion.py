import tensorflow as tf

''' PSNR - Peak Signal-to-Noise Ratio '''
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

''' SSIM - Structural Similarity Index '''
def ssim_metric(y_true, y_pred):
    """
    Calculation of Structural Similarity Index (SSIM)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        SSIM value
    """

    # y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    # y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)

    # Calculate SSIM using TensorFlow's image comparison functions
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)

    return ssim_value