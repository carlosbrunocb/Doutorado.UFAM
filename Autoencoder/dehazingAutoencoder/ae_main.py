from image_handling.imagesHandling import *
from model.autoencoder_model import *
from model.ae_loss_function import *
from model.ae_metric_fucntion import *


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configures GPU memory allocation to not be used
def disable_gpu():
    # List all available GPU physical devices
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPU devices:", physical_devices)

    # If there are GPUs available, disable all of them
    if physical_devices:
        tf.config.set_visible_devices([], 'GPU')  # Disable all GPUs
    # Check visible devices after setup (only CPU should be visible)
    print("Devices visible after disabling GPU:", tf.config.list_physical_devices())

# Configures GPU memory allocation to be done as needed
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def main():
    # tf.config.run_functions_eagerly(True)
    # tf.config.experimental_run_functions_eagerly(True)
    ''' ----- disable GPU Setting memory allocation ----- '''
    disable_gpu()

    ''' ----- GPU Setting memory allocation ----- '''
    # configure_gpu()

    ''' ----- Image Loading Setting to training ----- '''
    # directory path
    train_path = 'dataset/hazy'
    train_gt_path = 'dataset/hazy_gt'
    test_path = 'dataset/validation_hazy'
    test_gt_path = 'dataset/validation_gt'

    # images size of dataset
    # height x width
    # [0] 128 x 128
    # [1] 256 x 256
    # [2] 512 x 512
    size_image = [128, 256, 512]
    idx_s = 0

    # normalization
    normalization = True

    # information about dataset setting
    log = True

    # percentage for validation
    p_val = 0.8

    print('----- Loading dataset -----')

    # Create numpy array from dataset to training
    train_dataset, train_gt_dataset = create_dataset(train_path, train_gt_path,
                                                     size_image[idx_s], size_image[idx_s],
                                                     normalized=normalization,
                                                     verbose=log)
    test_dataset, test_gt_dataset = create_dataset(test_path, test_gt_path,
                                                   size_image[idx_s], size_image[idx_s],
                                                   normalized=normalization,
                                                   verbose=log)

    # tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, train_gt_dataset))
    # tf_train_dataset = tf_train_dataset.shuffle(buffer_size=500)
    #
    # # Data splitting for training and validation
    # tf_train_dataset = tf_train_dataset.take(int(p_val * len(train_dataset)))
    # tf_valid_dataset = tf_train_dataset.skip(int(p_val * len(train_dataset)))



    print('----- Dataset loaded into numpy array -----')

    ''' ----- Print images loaded from dataset ----- '''

    # print_rgb_images_vector(train_dataset, train_dataset.shape[0]//5 + 1, 5)
    # print_rgb_images_vector(train_gt_dataset, train_gt_dataset.shape[0] // 5 + 1, 5)
    # print_rgb_images_vector(test_dataset, test_dataset.shape[0] // 5 + 1, 5)
    # print_rgb_images_vector(test_gt_dataset, test_gt_dataset.shape[0] // 5 + 1, 5)

    # plt.figure(1)
    # plt.imshow(train_dataset[100])
    #
    # plt.figure(2)
    # plt.imshow(train_gt_dataset[100])
    # plt.show()

    print('')

    ''' ----- Model Setting ----- '''
    # Input_shape settings
    input_size = train_dataset[0].shape

    # Number of filters
    # number_filters = (100, 50, 30, 50, 100) 
    # number_filters = (128, 64, 32, 64, 128) # testado
    # number_filters = (100, 75, 50, 75, 100)
    # number_filters = (100, 50, 100, 50, 100)
    # number_filters = (25, 50, 100, 50, 25)
    # number_filters = (125, 75, 100, 50, 150)
    # number_filters = (150, 75, 100, 75, 150)
    # number_filters = (75, 50, 75)
    number_filters = (32, 8)

    # filter size vector
    # filters_size_vector = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3)]
    # filters_size_vector = [(5, 5), (3, 3), (7, 7), (3, 3), (5, 5)]
    # filters_size_vector = [(7, 7), (5, 5), (3, 3), (5, 5), (7, 7)]
    # filters_size_vector = [(3, 3), (5, 5), (7, 7), (5, 5), (3, 3)]
    # filters_size_vector = [(5, 5), (7, 7), (9, 9), (5, 5), (3, 3)]
    # filters_size_vector = [(3, 3), (7, 7), (9, 9), (5, 5), (3, 3)] # testado
    # filters_size_vector = [(15, 15), (9, 9), (7, 7), (5, 5), (3, 3)]
    # filters_size_vector = [(3, 3), (5, 5), (5, 5), (3, 3)]
    # filters_size_vector = [(15, 15), (9, 9), (3, 3)]
    # filters_size_vector = [(3, 3), (5, 5), (3, 3)]
    filters_size_vector = [(3, 3), (5, 5)]


    print('----- Creating Autoencoder Model -----')

    ae_model = ae_model_2CL_2MP_1FC_2DL(input_shape=input_size,
                                        number_filters=number_filters,
                                        filter_size=filters_size_vector)

    # ae_model = ae_model_3CL_3MP_1FC_3DL(input_shape=input_size,
    #                                     number_filters=number_filters,
    #                                     filter_size=filters_size_vector)

    # ae_model = ae_model_4CL_3MP_1FC_4DL(input_shape=input_size,
    #                                     number_filters=number_filters,
    #                                     filter_size=filters_size_vector)

    # ae_model = ae_model_5CL_3DP_3MP_1EM(input_shape=input_size)
    # ae_model = ae_model_5CL_3DP_3MP_1EM(input_shape=input_size,
    #                                     number_filters=number_filters,
    #                                     filter_size=filters_size_vector)


    # ae_model = ae_model_5CL_3MP_1FC_3DL(input_shape=input_size,
    #                                     number_filters=number_filters,
    #                                     filter_size=filters_size_vector)

    # ae_model = ae_model_5CL_5MP_1EM(input_shape=input_size,
    #                                 number_filters=number_filters,
    #                                 filter_size=filters_size_vector)

    # ae_model = ae_model_5CL_5MP_1FC_3DL(input_shape=input_size,
    #                                     number_filters=number_filters,
    #                                     filter_size=filters_size_vector)

    ae_model.summary()

    print('')

    print('----- Setting parameter for training -----')

    # Losses functions
    mse_losses_vector = ['mse',  # 0
                         'mean_absolute_percentage_error',  # 1
                         'mean_squared_logarithmic_error',  # 2
                         'mean_absolute_error',  # 3
                         'mean_squared_error']  # 4

    losses_vector = ['cosine_similarity',  # 0
                     'log_cosh',  # 1
                     'squared_hinge']  # 2

    entropy_losses_vector = ['binary_crossentropy',  # 0
                             'categorical_crossentropy',  # 1
                             'sparse_categorical_crossentropy']  # 2
    
    learn_rate = 0.0001
    # learn_rate = 0.00001

    # optimizer
    opt_setup = keras.optimizers.Adam(learning_rate=learn_rate)  # Adam
    # opt_setup = keras.optimizers.RMSprop(learning_rate=0.001) # RMSprop
    # opt_setup = keras.optimizers.SGD(learning_rate=0.001) # SGD

    optimizer = opt_setup
    # loss = lambda y_true, y_pred: combined_loss_mse_pl_l2_v2(y_true, y_pred, ae_model, l2_lambda=0.01)
    # loss = lambda y_true, y_pred: combined_loss_mse_pl_l2_v1(y_true, y_pred, vgg_model = vgg_model, l2_lambda=0.01)
    # loss = combined_loss_mse_l2_wrapper(ae_model, l2_lambda=0.01)
    # loss = mse_losses_vector[2]
    loss = combined_MSE_SSIM_UIQI_loss
    # loss = mahalanobis_loss
    # loss = combined_loss_mse_pl_l2_v2_wrapper(ae_model)
    # loss = combined_loss_mse_pl_l2_v1_wrapper()
    # loss = combined_loss_mse_cc
    # loss = combined_loss_mae_kld


    metrics = ['mae', psnr_metric]
    # metrics = ['mae', psnr_metric, ssim_metric]

    ae_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=metrics)

    print(opt_setup)
    print(loss)
    print(metrics)

    print('----- Initializing training -----')
    print('.....')

    # epochs number
    n_epochs = 2

    # batch size
    n_batch = 16

    print(f'n_epochs = {n_epochs}')
    print(f'n_batch = {n_batch}')

    # tf_train_dataset = tf_train_dataset.batch(batch_size=n_batch)
    # tf_valid_dataset = tf_train_dataset.batch(batch_size=n_batch)


    historic = ae_model.fit(train_dataset,
                            train_gt_dataset,
                            epochs=n_epochs,
                            batch_size=n_batch,
                            shuffle=True,
                            # validation_data=(test_dataset, test_gt_dataset)
                            validation_split=0.2
                            )

    # historic = ae_model.fit(tf_train_dataset,
    #                         epochs=n_epochs,
    #                         validation_data=tf_valid_dataset
    #                         )

    ''' ----- Validation ----- '''

    predicted_img = ae_model.predict(test_dataset)

    ''' ----- Save images predicted ----- '''

    path_out = 'prediction_outputs/ae_model_2CL_2MP_1FC_2DL'
    # path_out = 'prediction_outputs/ae_model_3CL_3MP_1FC_3DL'
    # path_out = 'prediction_outputs/ae_model_4CL_3MP_1FC_4DL'
    # path_out = 'prediction_outputs/ae_model_5CL_3DP_3MP_1EM'
    # path_out = 'prediction_outputs/ae_model_5CL_3MP_1FC_3DL'
    # path_out = 'prediction_outputs/ae_model_5CL_5MP_1EM'
    # path_out = 'prediction_outputs/ae_model_5CL_5MP_1FC_3DL'

    
    # output_name = 'combined_loss_mse_pl_l2_v2_' + 'dehazing'
    # output_name = 'combined_loss_mse_l2_wrapper_' + 'dehazing'
    output_name = 'loss_mse_log_' + 'dehazing'
    # output_name = 'combined_loss_mae_kld_' + 'dehazing'
    # output_name = 'combined_loss_mse_cc_' + 'dehazing'

    save_predicted_images(predicted_img, path_out, output_name)

    output_val = 'loss_mse_log_' + 'hazing'

    save_predicted_images(test_dataset, path_out, output_val)

    print_rgb_images_vector(test_dataset, test_dataset.shape[0] // 5 + 1, 5)
    print_rgb_images_vector(test_gt_dataset, test_gt_dataset.shape[0] // 5 + 1, 5)
    print_rgb_images_vector(predicted_img, predicted_img.shape[0] // 5 + 1, 5)

    ''' ----- Graph Plotting for Training Analysis ----- '''
    ''' ----- MAE Metric ----- '''
    plt.figure(1)
    plt.plot(historic.history['mae'])
    plt.plot(historic.history['val_mae'])
    plt.title('MAE por épocas')
    plt.xlabel('épocas')
    plt.ylabel('MAE')
    plt.legend(['treino', 'validação'])
    try:
        plt.savefig(f'{path_out}/{output_name}_MAE_Graph.png')
    except Exception as e:
        print(f"Error saving image {output_name}_MAE_Graph.png: {e}")

    ''' ----- PSNR Metric ----- '''
    plt.figure(2)
    plt.plot(historic.history['psnr_metric'])
    plt.plot(historic.history['val_psnr_metric'])
    plt.title('PSNR por épocas')
    plt.xlabel('épocas')
    plt.ylabel('PSNR')
    plt.legend(['treino', 'validação'])
    try:
        plt.savefig(f'{path_out}/{output_name}_PSNR_Graph.png')
    except Exception as e:
        print(f"Error saving image {output_name}_PSNR_Graph.png: {e}")

    ''' ----- Loss Metric ----- '''
    plt.figure(3)
    plt.plot(historic.history['loss'])
    plt.plot(historic.history['val_loss'])
    plt.title('Perdas por épocas')
    plt.xlabel('épocas')
    plt.ylabel('loss')
    plt.legend(['treino', 'validação'])
    try:
        plt.savefig(f'{path_out}/{output_name}_Loss_Graph.png')
    except Exception as e:
        print(f"Error saving image {output_name}_Loss_Graph.png: {e}")

    plt.show()


if(__name__ == '__main__'):
    main()