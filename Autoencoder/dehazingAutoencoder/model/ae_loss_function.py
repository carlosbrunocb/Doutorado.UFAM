import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.applications import VGG16

''' ----- loss functions ----- '''

''' MSE + Regularization L2 '''
def combined_loss_mse_l2(y_true, y_pred, model, l2_lambda=0.01):
  '''
  Loss function combined MSE + regularization L2

  Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model: Trained model
        l2_lambda: Weight for L2 regularization

  Returns:
        Combined loss value

  '''

  #  Loss of Reconstruction (MSE)
  reconstruction_loss = K.mean(K.square(y_true - y_pred))

  # Loss L2 (Regularization)
  # l2_loss = l2_lambda * K.mean(K.square(model.trainable_weights))
  regularization = l2_lambda * K.sum(
      [K.sum(K.square(w)) for w in model.trainable_weights])

  # Combining losses (adjust weights as needed)
  return reconstruction_loss + regularization

def combined_loss_mse_l2_wrapper(model, l2_lambda=0.01):
  def loss_mse_l2(y_true, y_pred):
    return combined_loss_mse_l2(y_true, y_pred, model, l2_lambda)
  return loss_mse_l2


''' MSE + Categorical Crossentropy '''
def Funcombined_loss_mse_cc(y_true, y_pred):
  '''
  Loss function combined MSE + Categorical Crossentropy

  Args:
    y_true: Ground truth labels
    y_pred: Predicted labels

  Returns:
    reconstruction_loss + weight * classification_loss: Combined loss value

  '''

  weight = 0.5

  #  Loss of Reconstruction (MSE)
  reconstruction_loss = keras.backend.mean(
      K.square(y_true[0] - y_pred[0]))

  # Classification Loss (Categorical Crossentropy)
  classification_loss = K.categorical_crossentropy(y_true[1], y_pred[1])

  # Combining losses (adjust weights as needed)
  return reconstruction_loss + weight * classification_loss

''' MSE + KLD '''
def kullback_leibler_divergence(y_true, y_pred):
  """
  Calculation of Kullback-Leibler Divergence

  Args:
      y_true: Ground truth labels
      y_pred: Predicted labels

  Returns:
      KLD value

  """

  # Adding a small value to avoid log(0)
  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # avoiding log(0)

  return K.sum(y_true * K.log(y_true / y_pred))

def combined_loss_mae_kld(y_true, y_pred):
  """
  Combined loss function: MAE + KLD

  Args:
    y_true: Ground truth labels
    y_pred: Predicted labels

  Returns:
    w1*mae + w2*kld: Combined loss value
  """

  print(f"y_true: {y_true}")
  print(f"y_pred: {y_pred}")

  w1 = 1 # weight for mae
  w2 = 1 # weight for kld

  mae = K.mean(K.abs(y_true - y_pred))
  kld = kullback_leibler_divergence(y_true, y_pred)

  print(f"MAE: {mae}")
  print(f"KLD: {kld}")

  return w1*mae + w2*kld  # You can adjust the weights as needed

''' MSE + Perceptual Loss + L2 '''
# Loading VGG16 without a final classification layer
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))

def combined_loss_mse_pl_l2_v1(y_true, y_pred,
                               vgg_model=vgg16_model,
                               l2_lambda=0.01):
  '''
  Combined loss function: MSE + Perceptual Loss + L2

  Args:
    y_true: Ground truth labels
    y_pred: Predicted labels
    l2_lambda: Weight for L2 regularization

  Returns:
    w1 * reconstruction + w2 * perceptual + w3 * regularization: Combined loss value
  '''

  # Reconstruction Loss using MSE
  reconstruction = K.mean(K.square(y_true - y_pred))

  # Loss perceptual (opcional)
  # Você pode usar uma rede pré-treinada como VGG para calcular a perda perceptual
  perceptual = perceptual_loss(y_true, y_pred, vgg_model)

  # Regularization using predicted image
  regularization = l2_lambda * K.sum(K.square(y_pred))

  # Weights for each components
  w1 = 1.0  # MSE
  w2 = 0.1  # loss perceptual
  w3 = 0.01 # regularization

  # Retornando a perda combinada
  return w1 * reconstruction + w2 * perceptual + w3 * regularization

def combined_loss_mse_pl_l2_v2(y_true, y_pred,
                               model,
                               vgg_model=vgg16_model,
                               l2_lambda=0.01):
  '''
  Combined loss function: MSE + Perceptual Loss + L2

  Args:
    y_true: Ground truth labels
    y_pred: Predicted labels
    model: Trained model
    l2_lambda: Weight for L2 regularization

  Returns:
    w1 * reconstruction + w2 * perceptual + w3 * regularization: Combined loss value
  '''

  # Reconstruction Loss using MSE
  reconstruction = K.mean(K.square(y_true - y_pred))

  # Loss perceptual (opcional)
  # Você pode usar uma rede pré-treinada como VGG para calcular a perda perceptual
  perceptual = perceptual_loss(y_true, y_pred, vgg_model)

  # Regularization using weights from trained model
  # Loss L2 (Regularization)
  # l2_loss = l2_lambda * K.mean(K.square(model.trainable_weights))
  regularization = l2_lambda * K.sum(
      [K.sum(K.square(w)) for w in model.trainable_weights])

  # Weights for each components
  w1 = 1.0  # MSE
  w2 = 0.1  # loss perceptual
  w3 = 0.05 # regularization

  # Retornando a perda combinada
  return w1 * reconstruction + w2 * perceptual + w3 * regularization

def perceptual_loss(y_true, y_pred, vgg_model):
  """
  Perceptual loss function

  Args:
    y_true: Ground truth labels
    y_pred: Predicted labels
    vgg_model: The pre-trained VGG16 model used for feature extraction

  Returns:
    loss: Perceptual loss value
  """

  # Defining the new network to extract features from the conv4_3 layer
  feature_extractor = keras.Model(inputs=vgg_model.input,
                                  outputs=vgg_model.get_layer("block4_conv3").output)

  # Extract features from true and reconstructed images
  true_features = feature_extractor(y_true)
  pred_features = feature_extractor(y_pred)

  # Calculate loss by mean square error (quadratic difference)
  loss = K.mean(K.square(true_features - pred_features))
  return loss

def combined_loss_mse_pl_l2_v1_wrapper(vgg_model=vgg16_model, l2_lambda=0.01):
  def loss_mse_pl_l2_v1(y_true, y_pred):
    return combined_loss_mse_pl_l2_v1(y_true,
                                      y_pred,
                                      vgg_model,
                                      l2_lambda)
  return combined_loss_mse_pl_l2_v1

def combined_loss_mse_pl_l2_v2_wrapper(model, vgg_model=vgg16_model, l2_lambda=0.01):
  def loss_mse_pl_l2_v2(y_true, y_pred):
    return combined_loss_mse_pl_l2_v2(y_true,
                                      y_pred,
                                      model,
                                      vgg_model,
                                      l2_lambda)
  return combined_loss_mse_pl_l2_v2

# Mahalanobis distance
# Mahalanobis function
def mahalanobis_distance(image_true, image_pred):
  '''
  Calculating the Mahalanobis distance between two images using
  the difference between their inverse covariance matrix and
  the difference between their mean vector.

  Mahalanobis distance = sqrt((x - mu)^T * inv(cov) * (x - mu))
  :. to application: sqrt((mp - mt)^T * (inv(covP) - inv(covT)) * (mp - mt))

  Args:
    image_true: image (batch, H, W, D)
    image_pred: image (batch, H, W, D)

  Returns:
    mahalanobis_dist: Mahalanobis distance
  '''

  # Calculating covariance matrix and your inverse
  _, inv_cov_true = inverse_and_covariance_matrix(image_true)
  _, inv_cov_pred = inverse_and_covariance_matrix(image_pred)

  # channel average for each image in the batch
  mean_true = tf.reduce_mean(image_true, axis=(1, 2), keepdims=True)
  mean_pred = tf.reduce_mean(image_pred, axis=(1, 2), keepdims=True)

  # Subtract the mean_pred from the mean_true of each channel
  # for each image in the batch
  diff_mean = mean_pred - mean_true
  batch_size, height, width, channels = tf.shape(diff_mean)
  diff_mean = tf.reshape(diff_mean, [batch_size, height * width, channels])

  # Subtract the inv_cov_pred from the inv_cov_true for each of them in the batch
  diff_inv_cov = inv_cov_pred - inv_cov_true

  # Multiplication of the inverse covariance matrix by the mean difference vector
  # (inv(covP) - inv(covT)) * (mp - mt))
  mahalanobis_inner = tf.linalg.matmul(diff_inv_cov, diff_mean, transpose_b=True)

  # Multiplication of the mean difference vector by mahalanobis_inner result
  # (mp - mt)^T * [(inv(covP) - inv(covT)) * (mp - mt)]
  mahalanobis_dist = tf.linalg.matmul(diff_mean, mahalanobis_inner)

  # Norm L2 (Removing negative value)
  mahalanobis_dist = tf.linalg.norm(mahalanobis_dist, axis=-1)

  # mahalanobis distance
  mahalanobis_dist = tf.sqrt(mahalanobis_dist)

  return mahalanobis_dist

# Function to calculate the inverse and covariance matrix
def inverse_and_covariance_matrix(image, batch_size=32):
  '''
  Calculates image covariance based on color channel

  Args:
    image: image (batch, H, W, D)

  Returns:
    covariancia: image covariance matrix
  '''
  # Reshape a image to (batch, H * W, D)
  # Ex.: (1, 8, 8, 3) -> (1, 64, 3)
  batch_size, height, width, channels = tf.shape(image)
  image_reshaped = tf.reshape(image, [batch_size, height * width, channels])

  # Subtract the mean of each channel for each image in the batch
  mean = tf.reduce_mean(image_reshaped, axis=1, keepdims=True)
  image_centered = image_reshaped - mean  # Centralizing data

  # Calculate a covariance matrix for each image in the batch
  covariancia = tf.linalg.matmul(image_centered, image_centered,
                                 transpose_a=True) / tf.cast(height * width - 1, tf.float32)

  # Calculate an inverse covariance matrix for each covariance matrix in the batch
  # Adds a small value to ensure numerical stability
  inverse_covariance = tf.linalg.inv(covariancia + 1e-6 * tf.eye(channels))

  # covariance matrix (batch_size, D, D)
  # inverse_covariance (batch_size, D, D)
  return covariancia, inverse_covariance

# Mahalanobis distance based loss function
def mahalanobis_loss(y_true, y_pred):
  '''
  Mahalanobis distance based loss function

  Args:
  y_true: true labels
  y_pred: predicted labels

  Returns:
  loss: mahalanobis distance
  '''

  loss = mahalanobis_distance(y_true, y_pred)

  return loss

###### Combined Loss Function ######
# Loss Function  Structural Similarity Index measure (SSIM)
def ssim_loss(y_true, y_pred):
  """
  loss fuction based on SSIM (Structural Similarity Index).

  Args:
    y_true: reference image (groundtruth)
    y_pred: predicted image

  Returns:
    ssim_value: loss value (1 - ssim_value)
  """
  ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
  return 1 - ssim_value

# Loss Function Universal Image Quality Index (UIQI)
def uiqi_loss(y_true, y_pred, c1=1e-4):
  """
  loss function based on UIQI (Universal Image Quality Index).

  Args:
    y_true: reference image (groundtruth)
    y_pred: predicted image

  Returns:
    uiqi_value: loss value (1 - uiqi_value)
  """
  # Calculation of means, variances and covariances
  mu_x = tf.reduce_mean(y_true, axis=[1, 2, 3], keepdims=True)
  mu_y = tf.reduce_mean(y_pred, axis=[1, 2, 3], keepdims=True)

  sigma_x = tf.reduce_mean(tf.square(y_true - mu_x), axis=[1, 2, 3], keepdims=True)
  sigma_y = tf.reduce_mean(tf.square(y_pred - mu_y), axis=[1, 2, 3], keepdims=True)

  sigma_xy = tf.reduce_mean((y_true - mu_x) * (y_pred - mu_y), axis=[1, 2, 3], keepdims=True)

  # Calculating the numerator and denominator of UIQI
  numerator = (4 * mu_x * mu_y * sigma_xy) + (c1)
  denominator = (tf.square(mu_x) + tf.square(mu_y)) * (sigma_x + sigma_y) + (c1)

  uiqi = numerator / denominator

  loss = 1 - uiqi

  return tf.reduce_mean(loss)

# Loss Function Mean Squared Error Logarítmica (log MSE)
def log_mse_loss(y_true, y_pred):
  """
  loss function based on log MSE (Mean Squared Error Logarítmica).

  Args:
    y_true: reference image (groundtruth)
    y_pred: predicted image

  Returns:
    loss: loss value
  """
  y_true_log = tf.math.log(y_true + 1)
  y_pred_log = tf.math.log(y_pred + 1)

  # Calculate the mean square error (MSE) for each pixel.
  log_mse = tf.square(y_true - y_pred)
  loss = tf.reduce_mean(log_mse)

  return loss

# Combined Loss Function based on Log MSE, SSIM e UIQI
def combined_MSE_SSIM_UIQI_loss(y_true, y_pred):
  """
  combined loss function based on log MSE, SSIM e UIQI.

  Args:
    y_true: reference image (groundtruth)
    y_pred: predicted image

  Returns:
    loss: loss value
  """
  # proportion constants
  log_mse_k = 0.5
  ssim_k = 0.25
  uiqi_k = 0.25

  log_mse_loss_value = log_mse_k * (log_mse_loss(y_true, y_pred))
  ssim_loss_value = ssim_k * (ssim_loss(y_true, y_pred))
  uiqi_loss_value = uiqi_k * (uiqi_loss(y_true, y_pred))

  loss = log_mse_loss_value + ssim_loss_value + uiqi_loss_value

  return loss