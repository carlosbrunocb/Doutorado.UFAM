from tensorflow import keras
from tensorflow.keras import layers

from math_transformation.apl_activation_function import *
from model.layers_ann.apl_layers_model import *


# APL - Adaptive Parameter Learning
# Regressive Convolutional Neural Network (R-CNN)
def build_regression_cnn(depth=10, filters=(64, 64, 64), num_channels=1, kernel_size=3, clip_output=False):
    """
    Regressive Convolutional Neural Network Build

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
                    filters[0]: for the first Conv2D layer
                    filters[1]: for the Conv2D layers in the depth loop
                    filters[2]: for the Conv2D layers before MaxPooling blocks
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings
    :param clip_output: If True, output values will be clipped to the 'output_range'.

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

    predicted_params = layers.Dense(3, activation=linear_plus_eps, name='predicted_parameters')(fc_dropout_x)

    powerlaw_x = (PowerLawTransformLayer(clip_output=clip_output, name='dynamic_power_law_transform')
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
        3 * num_masks, activation=linear_plus_eps, name='predicted_parameters'
    )(fc_dropout_x))

    # Apply transformation with new masked layer
    powerlaw_x = (PowerLawTransformWithDepthMapMasksLayer(
        clip_output=False, name='dynamic_power_law_transform'
    )([input_image, predicted_params, input_mask]))

    apl = keras.models.Model(inputs=[input_image, input_mask],
                             outputs=[powerlaw_x, predicted_params],
                             name='AplPowerLawWithMask')

    return apl


# APL - Adaptive Parameter Learning
# APL-DF - Adaptive Parameter Learning to Dehazy Function
# Regressive Convolutional Neural Network (R-CNN)
# No use depth maps as input [NDM - No Depth Maps]
def build_dehaze_fuction_by_regression_cnn_ndm(
        depth=10,
        filters=(64, 64, 64),
        num_channels=1,
        kernel_size=3):
    """
    Regressive Convolutional Neural Network Builder by ROI Masks

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings

    :return apl: AplPowerLaw model
    """
    # inputs [batch_images]
    input_image = layers.Input(shape=(None, None, num_channels), name='input_images')  # (B, H, W, 3)

    # cnn block
    # parameters layer [a, b]
    cnn_x = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_x = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.Activation('relu')(cnn_x)

        if i % 3 == 2:
            cnn_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_par_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # full conected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_par_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.3)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    # output layer: 2 parameters (a, b)
    predicted_params = (layers.Dense(2, activation=linear_plus_eps,
                                     name='predicted_parameters')(fc_dropout_x))  # predicted_params = (B, 2)

    # depth map Layer [d(x)]
    cnn_y = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_y = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_y)
        cnn_y = layers.BatchNormalization()(cnn_y)
        cnn_y = layers.Activation('relu')(cnn_y)

        if i % 3 == 2:
            cnn_y = layers.Dropout(0.2)(cnn_y)

    # predicted_dm = (B, H, W, 1)
    predicted_dm = layers.Conv2D(1, kernel_size, activation='sigmoid', padding='same', name='predicted_dm')(cnn_y)

    # Apply the dehazy fuction using a, b and d(x) parameters found in the ANN of parameters
    # and ANN of depth maps
    dehaze_xy = (DehazeLayer(name='dehazy_function')
                 ([input_image, predicted_dm, predicted_params]))

    apl = keras.models.Model(inputs=[input_image],
                             outputs=[dehaze_xy, predicted_dm, predicted_params],
                             name='APLDehazyFunction')

    return apl


# APL - Adaptive Parameter Learning
# APL-DF+PF - Adaptive Parameter Learning to Dehazy Function and PowerLaw Function
# Regressive Convolutional Neural Network (R-CNN)
def build_apl_to_df_and_pf_by_rcnn(
        depth=10,
        filters=(64, 64, 64),
        num_channels=1,
        kernel_size=3,
        clip_output=False):
    """
    Regressive Convolutional Neural Network Builder by ROI Masks

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings
    :param clip_output: If True, output values will be clipped to the 'output_range'.

    :return apl: AplPowerLaw model
    """
    # inputs [batch_images]
    input_image = layers.Input(shape=(None, None, num_channels), name='input_images')  # (B, H, W, 3)

    # cnn block
    # parameters layer [a, b]
    cnn_x = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_x = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.Activation('relu')(cnn_x)

        if i % 3 == 2:
            cnn_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_par_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # full conected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_par_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.3)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    # output layer: 2 parameters (a, b)
    predicted_params_ab = (layers.Dense(2, activation=linear_plus_eps,
                                        name='predicted_parameters_ab')(fc_dropout_x))  # predicted_params = (B, 2)

    # depth map layer [d(x)]
    cnn_y = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_y = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_y)
        cnn_y = layers.BatchNormalization()(cnn_y)
        cnn_y = layers.Activation('relu')(cnn_y)

        if i % 3 == 2:
            cnn_y = layers.Dropout(0.2)(cnn_y)

    # predicted_dm = (B, H, W, 1)
    predicted_dm = layers.Conv2D(1, kernel_size, activation='sigmoid', padding='same',
                                 name='predicted_dm')(cnn_y)

    # Apply the dehazy fuction using a, b and d(x) parameters found in the ANN of parameters and ANN of depth maps
    # I(x) = (H(x) - A * (1.0 - t(x))) / t(x)
    # t(x) = e^(-B * d(x))
    dehaze_xy = (DehazeLayer(name='dehazy_function')
                 ([input_image, predicted_dm, predicted_params_ab]))

    # powerlaw layer [f(x) = alpha * (x + epsilon) ^ gamma];
    # a = alpha; e = epsilon; g = gamma
    cnn_z = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_z = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_z)
        cnn_z = layers.BatchNormalization()(cnn_z)
        cnn_z = layers.Activation('relu')(cnn_z)

        if i % 3 == 2:
            cnn_z = layers.MaxPooling2D((2, 2), padding='same')(cnn_z)

    # cnn + maxpooling layer
    cnn_z = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_z)
    cnn_par_z = layers.MaxPooling2D((2, 2), padding='same')(cnn_z)

    # full conected layer
    # flatten layer
    fc_flatten_z = layers.GlobalAveragePooling2D()(cnn_par_z)

    # dense layer
    fc_dense_z = layers.Dense(256, activation='relu')(fc_flatten_z)
    fc_dropout_z = layers.Dropout(0.5)(fc_dense_z)

    fc_dense_z = layers.Dense(128, activation='relu')(fc_dropout_z)
    fc_dropout_z = layers.Dropout(0.3)(fc_dense_z)

    fc_dense_z = layers.Dense(64, activation='relu')(fc_dropout_z)
    fc_dropout_z = layers.Dropout(0.2)(fc_dense_z)

    # output layer: 3 parameters (a = alpha, e = epsilon, g = gamma)
    predicted_params_aeg = layers.Dense(3, activation=linear_plus_eps,
                                        name='predicted_parameters_aeg')(fc_dropout_z)

    # Apply the powerlaw fuction using a = alpha, e = epsilon and g = gamma parameters found in the ANN of parameters
    # f(x) = alpha * (x + epsilon) ^ gamma
    powerlaw_zxy = (PowerLawTransformLayer(clip_output=clip_output, name='dynamic_power_law_transform')
                    ([dehaze_xy, predicted_params_aeg]))

    apl = keras.models.Model(inputs=[input_image],
                             outputs=[powerlaw_zxy, dehaze_xy, predicted_dm, predicted_params_ab, predicted_params_aeg],
                             name='APL')

    return apl
