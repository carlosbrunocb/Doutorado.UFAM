from image_handling.ft_image_handling import *
from model.ft_analysis import *
from model.ft_loss_function import *
from model.ft_metric_function import *
from model.ft_ae_model import *

from sklearn.model_selection import KFold

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
    disable_gpu()

    ''' ----- GPU Setting memory allocation ----- '''
    # configure_gpu()

    ''' ----- Loading images dataset ----- '''
    # drive_path = '/content/drive/MyDrive/BOOK_ICMLA/dataset'
    drive_path = 'results/apl_raw'
    dim = (512, 512)
    # dim = (256, 256)
    # dim = (128, 128)
    sp_dir = 'outside'
    gt_dir = 'outside/gt'

    sp_datase, gt_dataset = load_images_from_folder(drive_path, sp_dir, gt_dir,
                                                    width=dim[0], height=dim[1],
                                                    resize=True)

    # plt.figure(1)
    # plt.imshow(sp_datase[0])
    # plt.axis('off')
    #
    # plt.figure(2)
    # plt.imshow(gt_dataset[0])
    # plt.axis('off')
    #
    # plt.show()

    ''' ----- Model Setting ----- '''
    # Input_shape settings
    input_size = sp_datase[0].shape

    # Number of filters
    # number_filters = (1024, 512, 512, 1024)
    # number_filters = (512, 256, 256, 512)
    number_filters = (512, 128, 128, 512)
    # number_filters = (256, 256, 256, 256)
    # number_filters = (256, 128, 128, 256)
    # number_filters = (128, 64, 64, 128)
    # number_filters = (512, 256, 128, 64, 128, 256, 512)
    # number_filters = (100, 50, 100)
    # number_filters = (32, 8)

    # filter size vector
    # filters_size_vector = [(3, 3), (5, 5), (7, 7), (9, 9), (7, 7), (5, 5), (3, 3)]
    filters_size_vector = [(3, 3), (5, 5), (5, 5), (3, 3)]
    # filters_size_vector = [(3, 3), (5, 5), (3, 3)]
    # filters_size_vector = [(3, 3), (5, 5)]

    # number of folds
    n_folds = 2

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
    # loss = mse_losses_vector[0]  # mse
    # loss = mse_losses_vector[3]  # mae
    loss = mse_losses_vector[5]  # rmse
    # metrics = ['mse', psnr_metric]
    metrics = [psnr_metric]

    # epochs number
    n_epochs = 60

    # batch size
    n_batch = 8

    # EarlyStopping: Stops training when val_loss does not improve for 10 consecutive seasons.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # ReduceLROnPlateau: Reduces the learning rate when val_loss does not improve for 5 consecutive seasons.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    print("\n:::::::: Setting ::::::::")
    print(f'input_size: {input_size}')
    print(f'learning rate: {lr_rate_vt[idx_lr_rate]}')
    print(optimizer)
    print(loss)
    print(metrics)
    print(f'n_filters = {number_filters}')
    print(f's_filters = {filters_size_vector}')
    print(f'n_epochs = {n_epochs}')
    print(f'n_batch = {n_batch}')
    print(f'n_folds = {n_folds}')
    print(f'early_stopping = {early_stopping}')
    print(f'reduce_lr = {reduce_lr}')

    ''' ----- Training -----'''
    print("\n:::::::: Training ::::::::")
    print("Evaluating model...")
    scores, histories = list(), list()

    kfold = KFold(n_splits=n_folds, shuffle=True, )
    fold_number = 1

    img_train = sp_datase
    img_gt = gt_dataset

    print(f'img_train: {img_train.shape}')
    print(f'img_gt: {img_gt.shape}')



    # print(rmse_loss(img_gt, img_train))

    for train_ix, test_ix in kfold.split(img_train):
        print(f'----- Fold {fold_number} -----')

        # define model
        # model = cnn_model(input_size, number_filters, filters_size_vector)
        # model = cnn5_model(input_size, number_filters, filters_size_vector)
        model = cnn5t_model(input_size, number_filters, filters_size_vector)
        # model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # select rows for train and test
        train_x, train_y = img_train[train_ix], img_gt[train_ix]
        test_x, test_y = img_train[test_ix], img_gt[test_ix]

        print(f'train_x: {train_x.shape}')
        print(f'train_y: {train_y.shape}')
        print(f'test_x: {test_x.shape}')
        print(f'test_y: {test_y.shape}')

        history = model.fit(train_x,
                            train_y,
                            epochs=n_epochs,
                            batch_size=n_batch,
                            shuffle=True,
                            validation_data=(test_x, test_y),
                            callbacks=[early_stopping, reduce_lr],
                            verbose=1)

        # evaluate model
        mse_model, psnr_model = model.evaluate(test_x, test_y, verbose=1)
        print('MSE: %.3f' % (mse_model))
        print('PSNR: %.3f' % (psnr_model))

        # stores scores
        scores.append((mse_model, psnr_model))
        histories.append(history)

        opt_setup = keras.optimizers.Adam(learning_rate=lr_rate_vt[idx_lr_rate])  # Adam
        optimizer = opt_setup

        ''' ----- Prediction -----'''
        predicted_img = model.predict(test_x)
        # dir_name = 'results/ae_apl/fold_' + str(fold_number)
        dir_name = 'results/ae_apl/512x512'
        file_name_in = 'cnn5.1_in_sots_ft_f'
        file_name_res = 'cnn5.1_res_sots_ft_f'
        file_name_gt = 'cnn5.1_gt_sots_ft_f'
        file_name_fig = 'grafico_mse_psnr'

        save_predicted_images(test_x, dir_name, file_name_in)
        save_predicted_images(predicted_img, dir_name, file_name_res)
        save_predicted_images(test_y, dir_name, file_name_gt)

        fold_number += 1

        if fold_number > n_folds - 1:
            # learning curves
            summarize_diagnostics(histories, dir_name, file_name_fig)

            # summarize estimated performance
            # summarize_performance(scores)

            print(predicted_img.shape)
            plt.figure(1)
            plt.imshow(predicted_img[0])
            plt.axis('off')

            plt.figure(2)
            plt.imshow(img_gt[0])
            plt.axis('off')

            plt.show()
            break


if __name__ == '__main__':
    main()
