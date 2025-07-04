from tensorflow import keras
from tensorflow.keras import layers

from math_transformation.pl_activation_function import *
from model.layers_ann.pl_layers_model import *


# APL - Adaptive Parameter Learning
# Regressive Convolutional Neural Network (R-CNN)
def build_regression_cnn(depth=10, filters=(64, 64, 64), num_channels=1, kernel_size=3):
    """
    Regressive Convolutional Neural Network Build

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
                    filters[0]: for the first Conv2D layer
                    filters[1]: for the Conv2D layers in the depth loop
                    filters[2]: for the Conv2D layers before MaxPooling blocks
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings

    :return apl: AplPowerLaw model
    """
    input_layer = layers.Input(shape=(None, None, num_channels), name='input')
    cnn_x = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_layer)

    for i in range(depth - 2):
        cnn_x = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.Activation('relu')(cnn_x)

        if i % 3 == 2:
            cnn_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # cnn + maxpooling layer
    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_maxpol_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # full conected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_maxpol_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.3)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    predicted_params = layers.Dense(3, activation=shifted_relu, name='predicted_parameters')(fc_dropout_x)

    powerlaw_x = (PowerLawTransformLayer(clip_output=False, name='dynamic_power_law_transform')
                  ([input_layer, predicted_params]))
    apl = keras.models.Model(inputs=input_layer, outputs=[powerlaw_x, predicted_params], name='AplPowerLaw')

    return apl


# APL - Adaptive Parameter Learning
# APL-DMS - Adaptive Parameter Learning by Depth Map Sectors
# Regressive Convolutional Neural Network (R-CNN)
def build_regression_cnn_with_dm_mask(
        depth=10,
        filters=(64, 64, 64),
        num_channels=1,
        kernel_size=3,
        num_masks=3):
    """
    Regressive Convolutional Neural Network Builder by ROI Masks

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings
    :param num_masks: The number of regions (masks) - ROI.

    :return apl: AplPowerLaw model
    """
    # inputs [batch_images, batch_masks]
    input_image = layers.Input(shape=(None, None, num_channels), name='input')
    input_mask = layers.Input(shape=(num_masks, None, None), name='input_mask')  # (B, M, H, W)

    # cnn block
    cnn_x = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_x = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.Activation('relu')(cnn_x)

        if i % 3 == 2:
            cnn_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_maxpol_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # full conected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_maxpol_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.3)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    # output layer:3*M parameters (α, ε, γ for each mask)
    predicted_params = (layers.Dense(
        3*num_masks, activation=linear_plus_eps, name='predicted_parameters'
    )(fc_dropout_x))

    # Apply transformation with new masked layer
    powerlaw_x = (PowerLawTransformWithDepthMapMasksLayer(
        clip_output=False, name='dynamic_power_law_transform'
    )([input_image, predicted_params, input_mask]))

    apl = keras.models.Model(inputs=[input_image, input_mask],
                             outputs=[powerlaw_x, predicted_params],
                             name='AplPowerLawWithMask')

    return apl
