from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract

'''
Function: Autoencoder Model
    It consists of 3 convolutional layers, 3 max pooling layers and full conected layer.
    Full conected layer consists of 3 dense layers, 2 dropout layers, a flatten layer and
    a reshape layer.
'''
# Autoencoder Model
def build_ae(input_shape, number_filters, filter_size):
    """
    Autoencoder Convolutional model
      * encoded_LCn: Conv2D layer
      * encoded_LMPooln: MaxPooling2D layer
      * encoded_flatten: Flatten layer
      * encoded_dense: Dense layer
      * encoded_dropout: Dropout layer
      * encoded: Dense layer [embedding]
      * decoder_dropout: Dropout layer
      * decoder_dense: Dense layer
      * decoder_reshape: Reshape layer
      * decoder_LUPooln: UpSampling2D layer
      * decoder_LDCn: Conv2DTranspose layer
      * decoded: Conv2D layer [decoded]

    Args:
      input_shape: (height, width, channels)
      number_filters: (50, 30, 20, 30, 50)
      filter_size: [(3, 3), (3, 3), (3, 3)] [kernel]

    Returns:
      model: autoencoder model [TensorFlow.keras model object (DNN)]
    """

    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_lc1 = keras.layers.Conv2D(number_filters[0], filter_size[0], activation='relu',
                                      padding='same')(input_image)
    encoded_lm_pool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_lc1)

    # second convolutional layer
    encoded_lc2 = keras.layers.Conv2D(number_filters[1], filter_size[1], activation='relu',
                                      padding='same')(encoded_lm_pool1)
    encoded_lm_pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_lc2)

    # third convolutional layer
    encoded_lc3 = keras.layers.Conv2D(number_filters[2], filter_size[2], activation='relu',
                                      padding='same')(encoded_lm_pool2)
    encoded_lm_pool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_lc3)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_lm_pool3)

    # dense layer
    encoded_dense = keras.layers.Dense(encoded_flatten.shape[1], activation='relu')(encoded_flatten)

    # dropout layer
    encoded_dropout = keras.layers.Dropout(0.2)(encoded_dense)

    # dense layer [embbeding]
    encoded = keras.layers.Dense(encoded_flatten.shape[1], activation='relu')(encoded_dropout)

    # Decoder
    # dropout layer
    decoder_dropout = keras.layers.Dropout(0.2)(encoded)

    # dense layer
    decoder_dense = keras.layers.Dense(encoded_flatten.shape[1], activation='relu')(decoder_dropout)

    # end full conected layer
    # reshape layer
    decoder_reshape = keras.layers.Reshape((encoded_lm_pool3.shape[1],
                                            encoded_lm_pool3.shape[2],
                                            encoded_lm_pool3.shape[3]))(decoder_dense)

    # first deconvolutional layer
    decoder_lu_pool1 = keras.layers.UpSampling2D((2, 2))(decoder_reshape)
    decoder_ldc1 = keras.layers.Conv2DTranspose(number_filters[2], filter_size[2], activation='relu',
                                                padding='same')(decoder_lu_pool1)

    # second deconvolutional layer
    decoder_lu_pool2 = keras.layers.UpSampling2D((2, 2))(decoder_ldc1)
    decoder_ldc2 = keras.layers.Conv2DTranspose(number_filters[1], filter_size[1], activation='relu',
                                                padding='same')(decoder_lu_pool2)

    # third deconvolutional layer
    decoder_lu_pool3 = keras.layers.UpSampling2D((2, 2))(decoder_ldc2)
    decoder_ldc3 = keras.layers.Conv2DTranspose(number_filters[0], filter_size[0], activation='relu',
                                                padding='same')(decoder_lu_pool3)

    # output layer [decoded]
    decoded = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoder_ldc3)

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder


