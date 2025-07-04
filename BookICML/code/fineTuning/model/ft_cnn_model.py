from tensorflow import keras


# CNN - 5 layer
def build_cnn(input_shape, number_filters, filter_size):
    """
    Convolutional Neural Network model
      * layer_CNN_n: Conv2D layer
      * layer_MPool_n: MaxPooling2D layer
      * layer_UPool_n: UpSampling2D layer
      * layer_dropout: Dropout layer

    Args:
      input_shape: (height, width, channels)
      number_filters: (50, 50, 50, 50)
      filter_size: [(3, 3), (3, 3), (3, 3), (3, 3)] [kernel]

    Returns:
      model: CNN model [TensorFlow.keras model object (DNN)]
    """
    # image input
    input_image = keras.layers.Input(shape=input_shape)

    # first convolutional layer
    layer_cnn_1 = keras.layers.Conv2D(number_filters[0], filter_size[0],
                                      activation='relu', padding='same')(input_image)
    layer_m_pool_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_1)

    # second convolutional layer
    layer_cnn_2 = keras.layers.Conv2D(number_filters[1], filter_size[1],
                                      activation='relu', padding='same')(layer_m_pool_1)
    layer_m_pool_2 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_2)
    layer_dropout = keras.layers.Dropout(0.2)(layer_m_pool_2)

    # third convolutional layer
    layer_u_pool_1 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_3 = keras.layers.Conv2D(number_filters[2], filter_size[2],
                                      activation='relu', padding='same')(layer_u_pool_1)
    layer_dropout = keras.layers.Dropout(0.2)(layer_cnn_3)

    # fourth convolutional layer
    layer_u_pool_2 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_4 = keras.layers.Conv2D(number_filters[3], filter_size[3],
                                      activation='relu', padding='same')(layer_u_pool_2)

    # fifth
    layer_cnn_5 = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(layer_cnn_4)

    cnn = keras.models.Model(input_image, layer_cnn_5)

    return cnn


# CNN - 5.1 layer
def build_cnn5_model(input_shape, number_filters, filter_size):
    """
    Convolutional Neural Network model
      * layer_CNN_n: Conv2D layer
      * layer_MPool_n: MaxPooling2D layer
      * layer_UPool_n: UpSampling2D layer
      * layer_dropout: Dropout layer

    Args:
      input_shape: (height, width, channels)
      number_filters: (50, 50, 50, 50)
      filter_size: [(3, 3), (3, 3), (3, 3), (3, 3)] [kernel]

    Returns:
      model: CNN model [TensorFlow.keras model object (DNN)]
    """
    # image input
    input_image = keras.layers.Input(shape=input_shape)

    # first convolutional layer
    layer_cnn_1 = keras.layers.Conv2D(number_filters[0], filter_size[0],
                                      activation='relu', padding='same')(input_image)
    layer_m_pool_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_1)

    # second convolutional layer
    layer_cnn_2 = keras.layers.Conv2D(number_filters[1], filter_size[1],
                                      activation='relu', padding='same')(layer_m_pool_1)
    layer_m_pool_2 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_2)
    # layer_dropout = keras.layers.Dropout(0.2)(layer_m_pool_2)
    layer_dropout = keras.layers.Dropout(0.5)(layer_m_pool_2)

    # third convolutional layer
    layer_u_pool_1 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_3 = keras.layers.Conv2D(number_filters[2], filter_size[2],
                                      activation='relu', padding='same')(layer_u_pool_1)
    layer_dropout = keras.layers.Dropout(0.5)(layer_cnn_3)

    # fourth convolutional layer
    layer_u_pool_2 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_4 = keras.layers.Conv2D(number_filters[3], filter_size[3],
                                      activation='relu', padding='same')(layer_u_pool_2)

    # fifth
    layer_cnn_5 = keras.layers.Conv2D(input_shape[2], (1, 1), activation='sigmoid',
                                      padding='same')(layer_cnn_4)

    cnn = keras.models.Model(input_image, layer_cnn_5)

    return cnn


# CNN - 5 layer (3 conv e 2 deconv
def build_cnn5t(input_shape, number_filters, filter_size):
    """
    Convolutional Neural Network model
      * layer_CNN_n: Conv2D layer
      * layer_MPool_n: MaxPooling2D layer
      * layer_UPool_n: UpSampling2D layer
      * layer_dropout: Dropout layer

    Args:
      input_shape: (height, width, channels)
      number_filters: (50, 50, 50, ..., 50)
      filter_size: [(3, 3), (3, 3), (3, 3), ..., (3, 3)] [kernel]

    Returns:
      model: CNN model [TensorFlow.keras model object (DNN)]
    """
    # image input
    input_image = keras.layers.Input(shape=input_shape)

    # first convolutional layer
    layer_cnn_1 = keras.layers.Conv2D(number_filters[0], filter_size[0],
                                      activation='relu', padding='same')(input_image)
    layer_m_pool_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_1)

    # second convolutional layer
    layer_cnn_2 = keras.layers.Conv2D(number_filters[1], filter_size[1],
                                      activation='relu', padding='same')(layer_m_pool_1)
    layer_m_pool_2 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_2)
    layer_dropout = keras.layers.Dropout(0.5)(layer_m_pool_2)

    # third convolutional layer
    layer_u_pool_1 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_dcnn_1 = keras.layers.Conv2DTranspose(number_filters[2], filter_size[2],
                                                activation='relu', padding='same')(layer_u_pool_1)
    layer_dropout = keras.layers.Dropout(0.5)(layer_dcnn_1)

    # fourth convolutional layer
    layer_u_pool_2 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_dcnn_2 = keras.layers.Conv2DTranspose(number_filters[3], filter_size[3],
                                                activation='relu', padding='same')(layer_u_pool_2)
    layer_dropout = keras.layers.Dropout(0.2)(layer_dcnn_2)

    # fifth convolutional layer
    layer_cnn_3 = keras.layers.Conv2D(input_shape[2], (1, 1), activation='sigmoid',
                                      padding='same')(layer_dropout)

    cnn = keras.models.Model(input_image, layer_cnn_3)

    return cnn


# CNN - 5 layer + 1 FC layer
def build_cnn_fc(input_shape, number_filters, filter_size):
    """
    Convolutional Neural Network model
      * layer_CNN_n: Conv2D layer
      * layer_MPool_n: MaxPooling2D layer
      * layer_UPool_n: UpSampling2D layer
      * layer_dropout: Dropout layer
      * layer_flatten: Flatten layer
      * layer_dense: Dense layer
      * layer_reshape: Reshape layer

    Args:
      input_shape: (height, width, channels)
      number_filters: (50, 50, 50, 50)
      filter_size: (3, 3, 3, 3) [kernel]

    Returns:
      model: CNN model [TensorFlow.keras model object (DNN)]
    """
    # image input
    input_image = keras.layers.Input(shape=input_shape)

    # first convolutional layer
    layer_cnn_1 = keras.layers.Conv2D(number_filters[0], filter_size[0],
                                      activation='relu', padding='same')(input_image)
    layer_m_pool_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_1)

    # second convolutional layer
    layer_cnn_2 = keras.layers.Conv2D(number_filters[1], filter_size[1],
                                      activation='relu', padding='same')(layer_m_pool_1)
    layer_m_pool_2 = keras.layers.MaxPooling2D((2, 2), padding='same')(layer_cnn_2)
    layer_dropout = keras.layers.Dropout(0.2)(layer_m_pool_2)

    # third convolutional layer
    layer_u_pool_1 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_3 = keras.layers.Conv2D(number_filters[2], filter_size[2],
                                      activation='relu', padding='same')(layer_u_pool_1)
    layer_dropout = keras.layers.Dropout(0.2)(layer_cnn_3)

    # fourth convolutional layer
    layer_u_pool_2 = keras.layers.UpSampling2D((2, 2))(layer_dropout)
    layer_cnn_4 = keras.layers.Conv2D(number_filters[3], filter_size[3],
                                      activation='relu', padding='same')(layer_u_pool_2)
    # full conected layer
    # flatten layer
    layer_flatten = keras.layers.Flatten()(layer_cnn_4)

    # dense layer
    layer_dense = keras.layers.Dense(layer_flatten.shape[1], activation='relu')(layer_flatten)

    # dropout layer
    layer_dropout = keras.layers.Dropout(0.2)(layer_dense)

    # end full conected layer
    # reshape layer
    layer_reshape = keras.layers.Reshape((layer_cnn_4.shape[1],
                                          layer_cnn_4.shape[2],
                                          layer_cnn_4.shape[3]))(layer_dropout)

    # fifth
    layer_cnn_5 = keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid',
                                      padding='same')(layer_reshape)

    cnn = keras.models.Model(input_image, layer_cnn_5)

    return cnn
