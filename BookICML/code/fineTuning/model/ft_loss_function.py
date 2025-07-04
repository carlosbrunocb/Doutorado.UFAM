import tensorflow as tf
from tensorflow.keras import backend as K

''' Root Mean Squared Error (RMSE) '''
def rmse_loss(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between 
    the original and reconstructed image

    Arguments:
       y_true: Tensor of the original image (batch_size, height, width, channels).
       y_pred: Tensor of the reconstructed image (batch_size, height, width, channels)
    
    Returns:
       Scalar tensor representing the RMSE loss value.
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
