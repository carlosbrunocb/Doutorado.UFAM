import matplotlib.pyplot as plt
import numpy as np


# plot diagnostic learning curves
def summarize_diagnostics(histories, path_fig, name_fig):
    plt.figure(figsize=(11, 9))

    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Loss Function')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.legend()

        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('PSNR')
        plt.plot(histories[i].history['psnr_metric'], color='blue', label='train')
        plt.plot(histories[i].history['val_psnr_metric'], color='orange', label='test')
        plt.legend()

    plt.savefig(f"{path_fig}/{name_fig}.png")
    plt.show()


# plot diagnostic learning curves
def summarize_diagnostics_curves(history, path_fig, name_fig):

    keys_list = list(history.history.keys())
    print(f'keys_list = {keys_list}')

    # split the keys
    train_losses = [k for k in keys_list
                    if "loss" in k and not k.startswith("val")]
    train_metrics = [k for k in keys_list
                     if "loss" not in k and not k.startswith("val") and k != "learning_rate"]

    # Combine metrics with their corresponding val_metrics
    all_train_keys = train_losses + train_metrics
    n_plots = len(all_train_keys)

    # figure size
    plt.figure(figsize=(8, 4 * n_plots))

    for i, key in enumerate(all_train_keys):
        plt.subplot(n_plots, 1, i + 1)
        plt.title(key.replace("_", " ").upper())
        plt.plot(history.history[key], label='train', color='blue')

        # Procurar a chave correspondente de validação, se existir
        val_key = f"val_{key}"
        if val_key in history.history:
            plt.plot(history.history[val_key], label='val', color='orange')

        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{path_fig}/{name_fig}.png")
    plt.show()

    # print(f'keys_list = {keys_list}')
    # plt.figure(figsize=(11, 9))
    #
    # # plot loss
    # plt.subplot(2, 1, 1)
    # plt.title('LOSS')
    # plt.plot(history.history[keys_list[1]], color='blue', label='train')
    # plt.plot(history.history[keys_list[3]], color='orange', label='test')
    # plt.legend()
    #
    # # plot accuracy
    # plt.subplot(2, 1, 2)
    # plt.title('PSNR')
    # plt.plot(history.history[keys_list[0]], color='blue', label='train')
    # plt.plot(history.history[keys_list[2]], color='orange', label='test')
    # plt.legend()
    #
    # plt.savefig(f"{path_fig}/{name_fig}.png")
    # plt.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores), np.std(scores), len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
