import matplotlib.pyplot as plt

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
    ''' ----- disable GPU Setting memory allocation ----- '''
    disable_gpu()

    ''' ----- GPU Setting memory allocation ----- '''
    # configure_gpu()

    ''' ----- Image Loading Setting to training ----- '''
    # directory path
    train_path = 'dataset/hazy'
    train_gt_path = 'dataset/hazy_gt'
    test_path = 'dataset/validation_gt'
    test_gt_path = 'dataset/validation_hazy'

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

    print('----- Dataset loaded into numpy array -----')

    ''' ----- Print images loaded from dataset ----- '''

    # print_rgb_images_vector(train_dataset, train_dataset.shape[0]//5 + 1, 5)
    # print_rgb_images_vector(train_gt_dataset, train_gt_dataset.shape[0] // 5 + 1, 5)
    # print_rgb_images_vector(test_dataset, test_dataset.shape[0] // 5 + 1, 5)
    # print_rgb_images_vector(test_gt_dataset, test_gt_dataset.shape[0] // 5 + 1, 5)

    print('')

    ''' ----- Model Setting ----- '''
    # Input_shape settings
    input_size = train_dataset[0].shape

    # Number of filters
    number_filters = (100, 50, 30, 50, 100)
    # number_filters = (100, 75, 50, 75, 100)
    # number_filters = (100, 50, 100, 50, 100)
    # number_filters = (25, 50, 100, 50, 25)
    # number_filters = (125, 75, 100, 50, 150)
    # number_filters = (150, 75, 100, 75, 150)
    # number_filters = (75, 50, 75)

    # filter size vector
    filters_size_vector = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3)]
    # filters_size_vector = [(5, 5), (3, 3), (7, 7), (3, 3), (5, 5)]
    # filters_size_vector = [(7, 7), (5, 5), (3, 3), (5, 5), (7, 7)]
    # filters_size_vector = [(3, 3), (5, 5), (7, 7), (5, 5), (3, 3)]
    # filters_size_vector = [(5, 5), (7, 7), (9, 9), (5, 5), (3, 3)]
    # filters_size_vector = [(3, 3), (7, 7), (9, 9), (5, 5), (3, 3)]
    # filters_size_vector = [(15, 15), (9, 9), (7, 7), (5, 5), (3, 3)]
    # filters_size_vector = [(15, 15), (9, 9), (3, 3)]

    print('----- Creating Autoencoder Model -----')

    ae_model = ae_model_3CL_3MP_1FC_3DL(input_shape=input_size,
                                        number_filters=number_filters,
                                        filter_size=filters_size_vector)

    # ae_model = ae_model_5CL_3DP_3MP_1EM(input_shape=input_size)

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

    # optimizer
    opt_setup = keras.optimizers.Adam(learning_rate=0.01)  # Adam
    # opt_setup = keras.optimizers.RMSprop(learning_rate=0.001) # RMSprop
    # opt_setup = keras.optimizers.SGD(learning_rate=0.001) # SGD

    optimizer = opt_setup
    # loss = lambda y_true, y_pred: combined_loss_mse_pl_l2_v2(y_true, y_pred, ae_model, l2_lambda=0.01)
    # loss = lambda y_true, y_pred: combined_loss_mse_pl_l2_v1(y_true, y_pred, vgg_model = vgg_model, l2_lambda=0.01)
    loss = combined_loss_mse_l2_wrapper(ae_model, l2_lambda=0.01)
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

    n_epochs = 100
    n_batch = 32

    historic = ae_model.fit(train_dataset,
                            train_gt_dataset,
                            epochs=n_epochs,
                            batch_size=n_batch,
                            shuffle=True,
                            # validation_data=(test_dataset, test_gt_dataset)
                            validation_split=0.2
                            )

    ''' ----- Validation ----- '''

    predicted_img = ae_model.predict(test_dataset)

    ''' ----- Save images predicted ----- '''

    # path_out = 'prediction_outputs/ae_model_3CL_3MP_1FC_3DL'
    path_out = 'prediction_outputs/ae_model_5CL_3DP_3MP_1EM'
    # path_out = 'prediction_outputs/ae_model_5CL_3MP_1FC_3DL'
    # path_out = 'prediction_outputs/ae_model_5CL_5MP_1EM'
    # path_out = 'prediction_outputs/ae_model_5CL_5MP_1FC_3DL'



    # output_name = 'combined_loss_mse_pl_l2_v2_' + 'dehazing'
    output_name = 'combined_loss_mse_l2_wrapper_' + 'dehazing'
    # output_name = 'combined_loss_mae_kld_' + 'dehazing'
    # output_name = 'combined_loss_mse_cc_' + 'dehazing'

    save_predicted_images(predicted_img, path_out, output_name)

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
    plt.savefig(f'{path_out}/{output_name}_MAE_Graph.png')

    ''' ----- PSNR Metric ----- '''
    plt.figure(2)
    plt.plot(historic.history['psnr_metric'])
    plt.plot(historic.history['val_psnr_metric'])
    plt.title('PSNR por épocas')
    plt.xlabel('épocas')
    plt.ylabel('PSNR')
    plt.legend(['treino', 'validação'])
    plt.savefig(f'{path_out}/{output_name}_PSNR_Graph.png')

    ''' ----- Loss Metric ----- '''
    plt.figure(3)
    plt.plot(historic.history['loss'])
    plt.plot(historic.history['val_loss'])
    plt.title('Perdas por épocas')
    plt.xlabel('épocas')
    plt.ylabel('loss')
    plt.legend(['treino', 'validação'])
    plt.savefig(f'{path_out}/{output_name}_Loss_Graph.png')

    plt.show()


if(__name__ == '__main__'):
    main()