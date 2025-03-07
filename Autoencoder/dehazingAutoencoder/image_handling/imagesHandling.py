import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

'''
Function: load_image
    It's used to load images from a specific directory path
    and aplly resizing on them.
'''
def load_image(image_path, width, height):
    '''
    Read image from path using tensorflow struct

    Args:
        image_path: path of image

    Returns:
        img: image
    '''

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img,
                                   height, width,
                                   method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, tf.uint8)

    return img

'''
Function: create_dataset
    It returns two images list of numpy array type:
      + dataset_ihazy: images list of numpy array
      + dataset_gt: images groundtruth list of numpy array
'''
def create_dataset(images_dir, gt_dir,
                   width, height,
                   normalized=False, verbose=True):
    '''
    Create numpy array from images dataset using tensorflow function
    to read and resize images.

    Args:
        images_dir: path of input images
        gt_dir: path of groundtruth images
        width, height: image sizing
        normalized: normalize the images in the range 0 to 1.
        verbose: show loading message

    Returns:
        dataset_ihazy: images list of numpy array
        dataset_gt: images groundtruth list of numpy array
    '''

    # List of all input images and their groundtruths
    images = sorted([os.path.join(images_dir, fname)
                     for fname in os.listdir(images_dir)
                     if (fname.endswith('.jpg') or fname.endswith('.JPG')
                         or fname.endswith('.png'))])

    gts = sorted([os.path.join(gt_dir, fname)
                  for fname in os.listdir(gt_dir)
                  if (fname.endswith('.jpg') or fname.endswith('.JPG')
                      or fname.endswith('.png'))])

    dataset_ihazy = []
    dataset_gt = []

    # Create a dataset using the images path list
    for img_path in images:
        image_hazy = load_image(img_path, width, height)
        # print(f'max\&min=({np.max(image_hazy)}, {np.min(image_hazy)})')
        dataset_ihazy.append(image_hazy.numpy().astype("uint8"))

    for gt_path in gts:
        image_gt = load_image(gt_path, width, height)
        dataset_gt.append(image_gt.numpy().astype("uint8"))

    dataset_ihazy = np.array(dataset_ihazy)
    dataset_gt = np.array(dataset_gt)

    if normalized:
        dataset_ihazy = dataset_ihazy / 255.0
        dataset_gt = dataset_gt / 255.0

    if verbose:
        print('\nData loaded')
        print(' - hazy images ')
        print(f'    size: {dataset_ihazy.shape}')
        print(f'    type: {type(dataset_ihazy)} {type(dataset_gt[0][0][0][0])}')
        print(f'    path: \n     {images}')

        print('------------------')

        print(' - groundtruth images')
        print(f'    size: {dataset_gt.shape}')
        print(f'    type: {type(dataset_gt)} {type(dataset_gt[0][0][0][0])}')
        print(f'    path: \n     {gts}')

    return dataset_ihazy, dataset_gt

def print_rgb_images_vector(img_vector, rows=2, columns=4):
    """
    Print all images inside in NumPy array as a grid.

    Args:
        img_vector: 4D Numpy array containing the colored images.
        rows: number of grid rows.
        columns: number of grid columns.
    """

    fig, axes = plt.subplots(rows, columns, figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < img_vector.shape[0]:
            ax.imshow(img_vector[i])
            ax.axis('off')
        else:
            ax.axis('off')  # hides the empty subplots

    plt.tight_layout()
    plt.show()


def save_predicted_images(images, path_out, exit_name):
    """
    Saves the images resulting from a model's prediction.

    Args:
        images: 4D images array NumPy.
        path_out: The directory where the images will be saved.
    """
    for i, imagem in enumerate(images):
        plt.imsave(f"{path_out}/{exit_name}_{i}.png", imagem)
