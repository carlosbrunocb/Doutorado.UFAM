from tensorflow import keras
from tensorflow.keras import layers

from math_transformation.ft_activation_function import shifted_relu
from model.ft_layers_ann.ft_layers_model import PowerLawTransformLayer


# APL - Adaptive Parameter Learning
# Regressive Convolutional Neural Network (R-CNN)
def build_regression_cnn(depth=10, filters=(64, 64, 64), num_channels=1, kernel_size=3):
    """
    Regressive Convolutional Neural Network Build

    Args:
        :param depth: network depth (controls the number of middle convolutional blocks)
        :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
                        filters[0]: for the first Conv2D layer
                        filters[1]: for the Conv2D layers in the depth loop
                        filters[2]: for the Conv2D layers before MaxPooling blocks
        :param num_channels: kernels settings (tuple of length 3 for initial, middle, and pooling blocks)
        :param kernel_size: kernels settings

    Returns:
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

    # first cnn + maxpooling layer
    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_maxpol_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # full conected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_maxpol_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    predicted_params = layers.Dense(3, activation=shifted_relu, name='predicted_parameters')(fc_dense_x)

    powerlaw_x = (PowerLawTransformLayer(clip_output=False, name='dynamic_power_law_transform')
                  ([input_layer, predicted_params]))
    apl = keras.models.Model(inputs=input_layer, outputs=powerlaw_x, name='AplPowerLaw')

    return apl
