from tensorflow import keras
from tensorflow.keras import layers

from model.layers_ann.ft_highboost_model import HighBoostLayer


# FT - Fine Tunning
# High-Boost Function
def build_highboost_cnn(
        depth=10,
        filters=(64, 64, 64),
        num_channels=1,
        kernel_size=3):
    """
    Regressive Convolutional Neural Network Builder to High-Boost filter

    :param depth: network depth (controls the number of middle convolutional blocks)
    :param filters: filters settings (tuple of length 3 for initial, middle, and pooling blocks)
    :param num_channels: number of input image channels (default=1)
    :param kernel_size: kernels settings

    :return hboost: High-Boost model
    """
    # inputs [batch_images, v_channel]
    input_image = layers.Input(shape=(None, None, num_channels), name='input_images')  # (B, H, W, 3)
    input_v_channel = layers.Input(shape=(None, None, 1), name='input_v_channel')  # (B, H, W, 1)

    # cnn block
    # parameters layer [a]
    cnn_x = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_image)

    for i in range(depth - 2):
        cnn_x = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.Activation('relu')(cnn_x)

        if i % 3 == 2:
            cnn_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    cnn_x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(cnn_x)
    cnn_par_x = layers.MaxPooling2D((2, 2), padding='same')(cnn_x)

    # fully connected layer
    # flatten layer
    fc_flatten_x = layers.GlobalAveragePooling2D()(cnn_par_x)

    # dense layer
    fc_dense_x = layers.Dense(256, activation='relu')(fc_flatten_x)
    fc_dropout_x = layers.Dropout(0.5)(fc_dense_x)

    fc_dense_x = layers.Dense(128, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.3)(fc_dense_x)

    fc_dense_x = layers.Dense(64, activation='relu')(fc_dropout_x)
    fc_dropout_x = layers.Dropout(0.2)(fc_dense_x)

    # output layer: predicted boost parameter 'a'
    predicted_params = (layers.Dense(1, activation='linear',
                                     name='predicted_parameters')(fc_dropout_x))  # predicted_params = (B, 1)

    # smoothing image Layer [s(x)]
    cnn_y = layers.Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_v_channel)

    for i in range(depth - 2):
        cnn_y = layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_y)
        cnn_y = layers.BatchNormalization()(cnn_y)
        cnn_y = layers.Activation('relu')(cnn_y)

        if i % 3 == 2:
            cnn_y = layers.Dropout(0.2)(cnn_y)

    # predicted_dm = (B, H, W, 1)
    predicted_v_blurred = layers.Conv2D(1, kernel_size, activation='sigmoid',
                                        padding='same', name='predicted_v_blurred')(cnn_y)

    # Apply the high-boost fuction using a, s(x) parameters found in the ANN of parameters
    hboost_xy = (HighBoostLayer(name='highboost_function')
                 ([input_v_channel, predicted_v_blurred, input_image, predicted_params]))

    hboost = keras.models.Model(inputs=[input_image, input_v_channel],
                                outputs=[hboost_xy, predicted_v_blurred, predicted_params],
                                name='FT-HighBoostFunction')

    return hboost
