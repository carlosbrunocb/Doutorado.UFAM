from tensorflow import keras
from tensorflow.keras import backend as K

'''
Function: Autoencoder Model
    It consists of 5 convolutional layers, 5 max pooling layers and embbeding layer
'''
def ae_model_5CL_5MP_1EM(input_shape,
                        number_filters = 16,
                        filter_size = 3
                        ):
    """
    Autoencoder Convolutional model
        * encoded_LCn: Conv2D layer
        * encoded_LMPooln: MaxPooling2D layer
        * encoded_flatten: Flatten layer
        * encoded: Dense layer [embedding]
        * decoder_dense: Dense layer
        * decoder_LUPooln: UpSampling2D layer
        * decoder_LDCn: Conv2DTranspose layer
        * decoded: Conv2D layer [decoded]

        Args:
            input_shape: (height, width, channels)
            number_filters: 16
            filter_size: 3 [kernel]

        Returns:
            model: autoencoder model [TensorFlow.keras model object (DNN)]
    """
    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_LC1 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(input_image)
    encoded_LMPool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC1)

    # second convolutional layer
    encoded_LC2 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool1)
    encoded_LMPool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC2)

    # third convolutional layer
    encoded_LC3 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool2)
    encoded_LMPool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC3)

    # fourth convolutional layer
    encoded_LC4 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool3)
    encoded_LMPool4 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC4)

    # fifth convolutional layer
    encoded_LC5 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool4)
    encoded_LMPool5 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC5)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_LMPool5)

    # dense layer [embbeding]
    encoded = keras.layers.Dense(encoded_flatten.shape[1], activation='relu')(encoded_flatten)

    # Decoder
    # full conected layer
    # dense layer
    decoder_dense = keras.layers.Reshape((encoded_LMPool5.shape[1], encoded_LMPool5.shape[2],
                                          encoded_LMPool5.shape[3]))(encoded)

    # first deconvolutional layer
    decoder_LUPool1 = keras.layers.UpSampling2D((2, 2))(decoder_dense)
    decoder_LDC1 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool1)

    # second deconvolutional layer
    decoder_LUPool2 = keras.layers.UpSampling2D((2, 2))(decoder_LDC1)
    decoder_LDC2 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool2)

    # third deconvolutional layer
    decoder_LUPool3 = keras.layers.UpSampling2D((2, 2))(decoder_LDC2)
    decoder_LDC3 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool3)

    # fourth deconvolutional layer
    decoder_LUPool4 = keras.layers.UpSampling2D((2, 2))(decoder_LDC3)
    decoder_LDC4 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool4)

    # fifth deconvolutional layer
    decoder_LUPool5 = keras.layers.UpSampling2D((2, 2))(decoder_LDC4)
    decoder_LDC5 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool5)

    # output layer [decoded]
    decoded = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoder_LDC5)

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder

'''
Function: Autoencoder Model
    It consists of 5 convolutional layers, 3 Dropout layers, 3 max pooling layers 
    and embbeding layer
'''
def ae_model_5CL_3DP_3MP_1EM(input_shape,
                             number_filters = 16,
                             filter_size = 3
                             ):
    """
    Autoencoder Convolutional model
        * encoded_LCn: Conv2D layer
        * encoded_LDropoutn: Droup layer
        * encoded_LMPooln: MaxPooling2D layer
        * encoded_flatten: Flatten layer
        * encoded: Dense layer [embedding]
        * decoder_dense: Dense layer
        * decoder_LUPooln: UpSampling2D layer
        * decoder_LDCn: Conv2DTranspose layer
        * decoded: Conv2D layer [decoded]

        Args:
            input_shape: (height, width, channels)
            number_filters: (50, 30, 20, 30, 50)
            filter_size: [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)] [kernel]

        Returns:
            model: autoencoder model [TensorFlow.keras model object (DNN)]
    """
    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_LC1 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(input_image)
    encoded_LDropout1 = keras.layers.Dropout(0.25)(encoded_LC1)
    encoded_LMPool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LDropout1)

    # second convolutional layer
    encoded_LC2 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool1)
    encoded_LMPool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC2)

    # third convolutional layer
    encoded_LC3 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool2)
    encoded_LDropout3 = keras.layers.Dropout(0.25)(encoded_LC3)
    encoded_LMPool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LDropout3)

    # fourth convolutional layer
    encoded_LC4 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool3)
    encoded_LDropout4 = keras.layers.Dropout(0.25)(encoded_LC4)
    encoded_LMPool4 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LDropout4)

    # fifth convolutional layer
    encoded_LC5 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool4)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_LC5)

    # dense layer [embbeding]
    encoded = keras.layers.Dense(encoded_flatten.shape[1], activation='relu')(encoded_flatten)

    # Decoder
    # full conected layer
    # dense layer
    decoder_dense = keras.layers.Reshape((encoded_LC5.shape[1], encoded_LC5.shape[2],
                                          encoded_LC5.shape[3]))(encoded)

    # first deconvolutional layer
    decoder_LDC5 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_dense)

    # second deconvolutional layer
    decoder_LUPool4 = keras.layers.UpSampling2D((2, 2))(decoder_LDC5)
    decoder_LDropout4 = keras.layers.Dropout(0.25)(decoder_LUPool4)
    decoder_LDC4 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LDropout4)

    # third deconvolutional layer
    decoder_LUPool3 = keras.layers.UpSampling2D((2, 2))(decoder_LDC4)
    decoder_LDropout3 = keras.layers.Dropout(0.25)(decoder_LUPool3)
    decoder_LDC3 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LDropout3)

    # fourth deconvolutional layer
    decoder_LUPool2 = keras.layers.UpSampling2D((2, 2))(decoder_LDC3)
    decoder_LDC2 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool2)

    # fifth deconvolutional layer
    decoder_LUPool1 = keras.layers.UpSampling2D((2, 2))(decoder_LDC2)
    decoder_LDropout1 = keras.layers.Dropout(0.25)(decoder_LUPool1)
    decoder_LDC1 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LDropout1)

    # output layer [decoded]
    decoded = (keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')
              (decoder_LDC1))

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder


'''
Function: Autoencoder Model
    It consists of 5 convolutional layers, 5 max pooling layers and full conected layer.
    Full conected layer consists of 3 dense layers, 2 dropout layers, a flatten layer and
    a reshape layer.
'''
def ae_model_5CL_5MP_1FC_3DL(input_shape,
                             number_filters = 16,
                             filter_size = 3
                            ):
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
        filter_size: [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)] [kernel]

    Returns:
        model: autoencoder model [TensorFlow.keras model object (DNN)]
    """

    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_LC1 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(input_image)
    encoded_LMPool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC1)

    # second convolutional layer
    encoded_LC2 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool1)
    encoded_LMPool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC2)

    # third convolutional layer
    encoded_LC3 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool2)
    encoded_LMPool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC3)

    # fourth convolutional layer
    encoded_LC4 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool3)
    encoded_LMPool4 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC4)

    # fifth convolutional layer
    encoded_LC5 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool4)
    encoded_LMPool5 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC5)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_LMPool5)

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
    decoder_reshape = keras.layers.Reshape((encoded_LMPool5.shape[1],
                                            encoded_LMPool5.shape[2],
                                            encoded_LMPool5.shape[3]))(decoder_dense)

    # first deconvolutional layer
    decoder_LUPool1 = keras.layers.UpSampling2D((2, 2))(decoder_reshape)
    decoder_LDC1 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool1)

    # second deconvolutional layer
    decoder_LUPool2 = keras.layers.UpSampling2D((2, 2))(decoder_LDC1)
    decoder_LDC2 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool2)

    # third deconvolutional layer
    decoder_LUPool3 = keras.layers.UpSampling2D((2, 2))(decoder_LDC2)
    decoder_LDC3 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool3)

    # fourth deconvolutional layer
    decoder_LUPool4 = keras.layers.UpSampling2D((2, 2))(decoder_LDC3)
    decoder_LDC4 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool4)

    # fifth deconvolutional layer
    decoder_LUPool5 = keras.layers.UpSampling2D((2, 2))(decoder_LDC4)
    decoder_LDC5 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool5)

    # output layer [decoded]
    decoded = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoder_LDC5)

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder

'''
Function: Autoencoder Model
    It consists of 5 convolutional layers, 3 max pooling layers and full conected layer.
    Full conected layer consists of 3 dense layers, 2 dropout layers, a flatten layer and
    a reshape layer.
'''
def ae_model_5CL_3MP_1FC_3DL(input_shape,
                             number_filters = 16,
                             filter_size = 3
                            ):
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
        number_filters: 16
        filter_size: 3 [kernel]

    Returns:
        model: autoencoder model [TensorFlow.keras model object (DNN)]
    """

    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_LC1 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(input_image)
    encoded_LMPool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC1)

    # second convolutional layer
    encoded_LC2 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool1)
    encoded_LMPool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC2)

    # third convolutional layer
    encoded_LC3 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool2)
    encoded_LMPool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC3)

    # fourth convolutional layer
    encoded_LC4 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool3)

    # fifth convolutional layer
    encoded_LC5 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LC4)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_LC5)

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
    decoder_reshape = keras.layers.Reshape((encoded_LC5.shape[1],
                                            encoded_LC5.shape[2],
                                            encoded_LC5.shape[3]))(decoder_dense)

    # first deconvolutional layer
    decoder_LDC1 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_reshape)

    # second deconvolutional layer
    decoder_LDC2 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LDC1)

    # third deconvolutional layer
    decoder_LUPool3 = keras.layers.UpSampling2D((2, 2))(decoder_LDC2)
    decoder_LDC3 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool3)

    # fourth deconvolutional layer
    decoder_LUPool4 = keras.layers.UpSampling2D((2, 2))(decoder_LDC3)
    decoder_LDC4 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool4)

    # fifth deconvolutional layer
    decoder_LUPool5 = keras.layers.UpSampling2D((2, 2))(decoder_LDC4)
    decoder_LDC5 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool5)

    # output layer [decoded]
    decoded = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoder_LDC5)

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder

'''
Function: Autoencoder Model
    It consists of 3 convolutional layers, 3 max pooling layers and full conected layer.
    Full conected layer consists of 3 dense layers, 2 dropout layers, a flatten layer and
    a reshape layer.
'''
def ae_model_3CL_3MP_1FC_3DL(input_shape,
                             number_filters=16,
                             filter_size=3
                            ):
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
      number_filters: 16
      filter_size: 3 [kernel]

    Returns:
      model: autoencoder model [TensorFlow.keras model object (DNN)]
    """

    input_image = keras.layers.Input(shape=input_shape)

    # Encoder
    # first convolutional layer
    encoded_LC1 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(input_image)
    encoded_LMPool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC1)

    # second convolutional layer
    encoded_LC2 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool1)
    encoded_LMPool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC2)

    # third convolutional layer
    encoded_LC3 = keras.layers.Conv2D(number_filters, filter_size, activation='relu',
                                      padding='same')(encoded_LMPool2)
    encoded_LMPool3 = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded_LC3)

    # full conected layer
    # flatten layer
    encoded_flatten = keras.layers.Flatten()(encoded_LMPool3)

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
    decoder_reshape = keras.layers.Reshape((encoded_LMPool3.shape[1],
                                            encoded_LMPool3.shape[2],
                                            encoded_LMPool3.shape[3]))(decoder_dense)

    # first deconvolutional layer
    decoder_LUPool1 = keras.layers.UpSampling2D((2, 2))(decoder_reshape)
    decoder_LDC1 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool1)

    # second deconvolutional layer
    decoder_LUPool2 = keras.layers.UpSampling2D((2, 2))(decoder_LDC1)
    decoder_LDC2 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool2)

    # third deconvolutional layer
    decoder_LUPool3 = keras.layers.UpSampling2D((2, 2))(decoder_LDC2)
    decoder_LDC3 = keras.layers.Conv2DTranspose(number_filters, filter_size, activation='relu',
                                                padding='same')(decoder_LUPool3)

    # output layer [decoded]
    decoded = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoder_LDC3)

    autoencoder = keras.models.Model(input_image, decoded)

    return autoencoder

