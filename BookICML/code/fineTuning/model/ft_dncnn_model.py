from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, Multiply

from model.ft_layers_ann.ft_layers_model import *

'''
Function: Denoising Convolutional Neural Model
    CNN network with depth 17
'''


# DnCnn - Denoising Convolutional Neural Network Build
def build_dncnn(depth=17, filters=(64, 64), num_channels=1, kernel_size=(3, 3)):
    """
    Denoising Convolutional Neural Network Build
      * CNN_n: Conv2D layer

    Args:
        :param depth: network depth
        :param filters: filters settings (tuple)
        :param num_channels: number of channels
        :param kernel_size: kernels settings (tuple)

    Returns:
        :return dncnn: DnCNN model
    """
    input_layer = Input(shape=(None, None, num_channels), name='input')
    cnn_x = Conv2D(filters[0], kernel_size[0], padding='same', activation='relu')(input_layer)

    for _ in range(depth - 2):
        cnn_x = keras.layers.Conv2D(filters[1], kernel_size[1], padding='same')(cnn_x)
        cnn_x = BatchNormalization()(cnn_x)
        cnn_x = Activation('relu')(cnn_x)

    cnn_x = Conv2D(num_channels, kernel_size[1], padding='same')(cnn_x)

    # Denoised image = input - predicted noise
    output_layer = Subtract()([input_layer, cnn_x])

    dncnn = keras.models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')

    return dncnn


# DnCnn - Denoising Convolutional Neural Network Build with Color Adjustment
def build_dncnn_adj_color(depth=(17, 10), filters=(64, 64, 64, 64),
                          num_channels=1, kernel_size=(3, 3)):
    """
    Denoising Convolutional Neural Network Build
    + Color Adjustment
      * CNN_n: Conv2D layer

    Args:
        :param depth: network depth
        :param filters: filters settings (tuple)
        :param num_channels: number of channels
        :param kernel_size: kernels settings (tuple)

    Returns:
        :return dncnn: DnCNN model
    """
    input_layer = Input(shape=(None, None, num_channels), name='input')
    cnn_x = Conv2D(filters[0], kernel_size, padding='same', activation='relu')(input_layer)

    for _ in range(depth[0] - 2):
        cnn_x = keras.layers.Conv2D(filters[1], kernel_size, padding='same')(cnn_x)
        cnn_x = BatchNormalization()(cnn_x)
        cnn_x = Activation('relu')(cnn_x)

    cnn_x = Conv2D(num_channels, kernel_size, padding='same')(cnn_x)

    # Denoised image = input - predicted noise
    dncnn = Subtract()([input_layer, cnn_x])

    cnn_y = Conv2D(filters[2], kernel_size, padding='same', activation='relu')(dncnn)

    for _ in range(depth[1] - 2):
        cnn_y = keras.layers.Conv2D(filters[3], kernel_size, padding='same')(cnn_y)
        cnn_y = BatchNormalization()(cnn_y)
        cnn_y = Activation('relu')(cnn_y)

    cnn_y = Conv2D(num_channels, kernel_size, padding='same')(cnn_y)

    # Adjustment color = Denoised image * predicted (Hadamard product)
    adjusted_color_output = Multiply()([dncnn, cnn_y])
    cnn_y = Conv2D(num_channels, kernel_size, padding='same')(adjusted_color_output)
    adjusted_color_output = Multiply()([input_layer, cnn_y])

    adcodncnn = keras.models.Model(inputs=input_layer, outputs=adjusted_color_output, name='AdCoDnCNN')

    return adcodncnn
