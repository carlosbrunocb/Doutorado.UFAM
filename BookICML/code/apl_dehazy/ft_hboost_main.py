from image_handling.apl_image_handling import *
from model.apl_analysis import *
from model.apl_apl_model import *
from depth_map.apl_depth_maps import *
from model.apl_loss_function import *
from model.apl_metric_function import *
from math_transformation.apl_transformation import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger

import logging

from model.ft_model import build_highboost_cnn


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
    """ ----- disable GPU Setting memory allocation ----- """
    # disable_gpu()

    ''' ----- GPU Setting memory allocation ----- '''
    configure_gpu()

    ''' ----- Loading images dataset ----- '''
    drive_path = 'results/dh_pl/dehazy_function/512x512'
    sp_dir = 'res'
    gt_dir = 'gt'

    dim = (512, 512)
    # dim = (256, 256)
    # dim = (128, 128)

    sp_dataset, gt_dataset = load_images_from_folder(drive_path,
                                                     sp_dir,
                                                     gt_dir,
                                                     width=dim[0], height=dim[1],
                                                     resize=True)
    v_dataset = rgb_to_v_channel(sp_dataset)

    print(f"sp_dataset = {sp_dataset.shape}")
    print(f"gt_dataset = {gt_dataset.shape}")
    print(f"v_dataset = {v_dataset.shape}")

    # # Print inputs
    # for i, _ in enumerate(sp_dataset):
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(sp_dataset[i])
    #     plt.title("IN")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(gt_dataset[i])
    #     plt.title("GT")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(v_dataset[i])
    #     plt.title("V")
    #     plt.axis('off')
    #
    #     plt.show()

    ''' ----- Model Setting ----- '''
    # Input_shape settings
    input_size = sp_dataset[0].shape

    # Number of filters
    # number_filters = (1024, 512, 1024)
    # number_filters = (128, 512, 1024)
    # number_filters = (512, 256, 512)
    # number_filters = (256, 128, 256)
    # number_filters = (128, 64, 128)
    number_filters = (64, 64, 64)

    # filter size vector
    filters_size = (3, 3)

    # network depth
    n_depth = 12

    # split percentage
    s_perc = 0.2

    # Losses functions
    mse_losses_vector = ['mse',  # 0
                         'mean_absolute_percentage_error',  # 1
                         'mean_squared_logarithmic_error',  # 2
                         'mean_absolute_error',  # 3
                         'mean_squared_error',  # 4
                         rmse_loss]  # 5

    # learning rate setting
    lr_rate_vt = [0.01, 0.001, 0.0001]
    idx_lr_rate = 2

    # optimizer
    opt_setup = keras.optimizers.Adam(learning_rate=lr_rate_vt[idx_lr_rate])  # Adam

    # training parameters setting
    optimizer = opt_setup
    loss = mse_losses_vector[0]  # mse
    # loss = mse_losses_vector[3]  # mae
    # loss = mse_losses_vector[5]  # rmse
    metrics = [psnr_metric]

    # epochs number
    n_epochs = 80

    # batch size
    n_batch = 8

    # EarlyStopping: Stops training when val_loss does not improve for 10 consecutive seasons.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # ReduceLROnPlateau: Reduces the learning rate when val_loss does not improve for 5 consecutive seasons.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    # Setting ANN Model
    # model_choice = 'regression_cnn'
    # model_choice = 'regression_cnn_with_dm_mask'
    # model_choice = 'apl_to_dehaze_fuction_by_regression_cnn'
    model_choice = 'highboost_cnn'

    # ANN Model (architecture)
    model_builders = {
        'regression_cnn': build_regression_cnn,
        'regression_cnn_with_dm_mask': build_regression_cnn_with_dm_mask,
        'dehaze_fuction_by_regression_cnn_ndm': build_dehaze_fuction_by_regression_cnn_ndm,
        'highboost_cnn': build_highboost_cnn
    }

    print("\n:::::::: Setting ::::::::")
    print(f'input_size: {input_size}')
    print(f'learning rate: {lr_rate_vt[idx_lr_rate]}')
    print(optimizer)
    print(loss)
    print(metrics)
    print(f'n_filters = {number_filters}')
    print(f's_filters = {filters_size}')
    print(f'n_epochs = {n_epochs}')
    print(f'n_batch = {n_batch}')
    print(f's_perc = {s_perc}')
    print(f'n_depth = {n_depth}')
    print(f'early_stopping = {early_stopping}')
    print(f'reduce_lr = {reduce_lr}')
    print(f'ANN Model choice: {model_choice}')

    ''' ----- Training -----'''
    print("\n:::::::: Training ::::::::")
    print("Input Dataset ...")

    img_train = sp_dataset
    img_gt = gt_dataset
    img_vb = v_dataset.numpy()

    print(f'img_train: {img_train.shape}')
    print(f'img_gt: {img_gt.shape}')
    print(f'img_vb: {img_vb.shape}')
    print(f'type img_train: {img_train.dtype}')
    print(f'type img_gt: {img_gt.dtype}')
    print(f'type img_vb: {img_vb.dtype}')

    num_idx = np.arange(len(sp_dataset))

    train_ix, test_ix = train_test_split(num_idx, test_size=s_perc, random_state=42)

    # select rows for train and test
    train_x, train_y, train_v = img_train[train_ix], img_gt[train_ix], img_vb[train_ix]
    test_x, test_y, test_v = img_train[test_ix], img_gt[test_ix], img_vb[test_ix]

    # for i, _ in enumerate(train_x):
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(train_x[i])
    #     plt.title("IN")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(train_y[i])
    #     plt.title("GT")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(train_v[i])
    #     plt.title("VB")
    #     plt.axis('off')
    #
    #     plt.show()

    # Create a dummy dataset
    dummy_train_params = np.zeros((len(train_y), 1), dtype=np.float32)
    dummy_train_vblurred_params = np.zeros((len(train_y), dim[0], dim[1], 1), dtype=np.float32)
    dummy_test_params = np.zeros((len(test_y), 1), dtype=np.float32)
    dummy_test_vblurred_params = np.zeros((len(test_y), dim[0], dim[1], 1), dtype=np.float32)

    print(f"train_x = {train_x.shape}")
    print(f"train_y = {train_y.shape}")
    print(f"dummy_train_params = {dummy_train_params.shape}")
    print(f"dummy_train_vblurred_params = {dummy_train_vblurred_params.shape}")
    print(f"test_x = {test_x.shape}")
    print(f"test_y = {test_y.shape}")
    print(f"dummy_test_params = {dummy_test_params.shape}")
    print(f"dummy_test_vblurred_params = {dummy_test_vblurred_params.shape}")

    # dir_name = 'results/dehaze/512x512'
    dir_name = 'results/dh_pl/fine_tunning/hb_512x512'
    # dir_name = 'results/dehaze_apl/512x512/ni_2/v2'
    file_name_in = 'in_apl_sots'
    file_name_res = 'res_apl_sots'
    file_name_vblurred = 'vblurred_ft_sots'
    file_name_gt = 'gt_apl_sots'
    file_name_fig = 'grafico_loss_psnr'

    print("\nTraining model...")
    print(f"train_x = {type(train_x)}")
    print(f"train_x = {type(train_x.dtype)}")
    print(f"dummy_train_params = {type(dummy_train_params)}")
    print(f"dummy_train_params = {type(dummy_train_params.dtype)}")
    print(f"dummy_train_vblurred_params = {type(dummy_train_vblurred_params)}")
    print(f"dummy_train_vblurred_params = {type(dummy_train_vblurred_params.dtype)}")

    # Log file in csv format
    csv_logger = CSVLogger(f'{dir_name}/training_log.csv', append=False)

    model = model_builders[model_choice](
        depth=n_depth,
        filters=number_filters,
        num_channels=img_train.shape[3],
        kernel_size=filters_size
    )

    model.summary()
    # model.compile(optimizer=optimizer,
    #               loss={'highboost_function': loss,
    #                     'predicted_v_blurred': None,  # Don't apply loss function
    #                     'predicted_parameters': None},  # Don't apply loss function
    #               metrics={'highboost_function': metrics})
    model.compile(
        optimizer=optimizer,
        loss={
            'highboost_function': loss,  # main loss
            'predicted_v_blurred': 'mse',  # loss dummy
            'predicted_parameters': 'mse'  # loss dummy
        },
        loss_weights={
            'highboost_function': 1.0,  # weigth 1 for main output
            'predicted_v_blurred': 0.0,  # weigth 0 for auxiliary output
            'predicted_parameters': 0.0  # weigth 0 for auxiliary output
        },
        metrics={'highboost_function': metrics}
    )
    history = model.fit([train_x, train_v],  # Inputs: image and v channel
                        # Outputs: transformed image and dummy parameters
                        [train_y, dummy_train_vblurred_params, dummy_train_params],
                        epochs=n_epochs,
                        batch_size=n_batch,
                        shuffle=True,
                        validation_data=([test_x, test_v], [test_y, dummy_test_vblurred_params, dummy_test_params]),
                        callbacks=[early_stopping, reduce_lr, csv_logger],
                        verbose=1)

    # evaluate model
    print("\nEvaluating model...")
    # results[0] is the loss ('dynamic_power_law_transform', which is the MSE).
    # results[1] is the first metric of the first output (PSNR).
    results = model.evaluate([test_x, test_v], [test_y, dummy_test_vblurred_params, dummy_test_params], verbose=1)
    print(f'results = {results}')
    loss_rate = results[0]  # average image MSE
    psnr_rate = results[-1]  # avarege image PSNR
    print('MSE: %.4f' % loss_rate)
    print('PSNR: %.4f' % psnr_rate)

    # Setting logging
    logging.basicConfig(
        filename=f'{dir_name}/evaluation_model.log',  # file name
        level=logging.INFO,  # Minimum level for logger (INFO)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with time, level and message
        datefmt='%Y-%m-%d %H:%M:%S'  # date/time format
    )

    logging.info('Evaluation Model:')
    logging.info('MSE: %.4f', loss_rate)
    logging.info('PSNR: %.4f', psnr_rate)
    logging.info('---')

    # save model
    try:
        model.save(f'{dir_name}/{model_choice}_model.keras')
    except Exception as e:
        print(f"Error trying to save the model: {e}")

    # learning curves
    summarize_diagnostics_curves(history, dir_name, file_name_fig)

    ''' ----- Prediction -----'''
    print("\nDenoising images (Predicting) ...")
    # predicted_img, predictions_dm, predictions_params = model.predict([test_x])
    predicted_img, predictions_vblurred, predictions_params = model.predict([img_train, img_vb])
    print(f':predicted_img max:{np.max(predicted_img)}')
    print(f':predicted_img min:{np.min(predicted_img)}')
    print(f':predictions_vblurred max:{np.max(predictions_vblurred)}')
    print(f':predictions_vblurred min:{np.min(predictions_vblurred)}')
    print(f':predictions_params max:{np.max(predictions_params)}')
    print(f':predictions_params min:{np.min(predictions_params)}')

    print(' ----- Parameters  Predicted ----- ')
    for par in predictions_params:
        print(par)

    # save prediction
    try:
        img_name_file = 'predicted_img'
        vbl_name_file = 'predictions_vblurred'
        par_name_file = 'predictions_params'
        np.save(f"{dir_name}/{img_name_file}.npy", predicted_img)
        np.save(f"{dir_name}/{vbl_name_file}.npy", predictions_vblurred)
        np.save(f"{dir_name}/{par_name_file}.npy", predictions_params)
        print(f'predicted image vector salved!\nname file: {img_name_file}.npy')
        print(f'predicted image vector salved!\nname file: {vbl_name_file}.npy')
        print(f'predicted parameters vector salved!\nname file: {par_name_file}.npy')
    except Exception as e:
        print(f"Error saving predicted image vector: {e}")

    # predicted_img = normalize_to_0_1(predicted_img)

    save_predicted_images(img_train, f'{dir_name}/in', file_name_in)
    save_predicted_images(predicted_img, f'{dir_name}/res', file_name_res)
    save_predicted_images(predictions_vblurred, f'{dir_name}/vblurred', file_name_vblurred)
    save_predicted_images(img_gt, f'{dir_name}/gt', file_name_gt)

    plt.figure(1)
    plt.imshow(img_train[0])
    plt.axis('off')

    plt.figure(2)
    plt.imshow(predicted_img[0])
    plt.axis('off')

    plt.figure(3)
    plt.imshow(img_gt[0])
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
