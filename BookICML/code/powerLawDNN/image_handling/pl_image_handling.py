import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


# Loads an image according to the desired settings
def load_image(image_path, width=128, height=128, resize=False,
               grayscale=False, normalized=False):
    """
    Read image from path using tensorflow struct

    Args:
      :param image_path: path of image
      :param width: width of image
      :param height: height of image
      :param resize: resize image
      :param grayscale: inform whether the image is in grayscale or color.
      :param normalized: If the value is True, then normalize the image [0.0, 1.0]

    Returns:
      :return img: image

    """
    img = tf.io.read_file(image_path)

    if grayscale:
        img = tf.image.decode_jpeg(img, channels=1)
    else:
        img = tf.image.decode_jpeg(img, channels=3)

    if resize:
        img = tf.image.resize_with_pad(img, height, width,
                                       method=tf.image.ResizeMethod.BICUBIC)

    if normalized:
        print("Normalizing")
        img = tf.cast(img, tf.uint8)
        img = tf.image.convert_image_dtype(img, tf.float32)
    else:
        img = tf.cast(img, tf.uint8)

    return img


# Loads a set of images from a folder.
def load_images_from_folder(folder_path,
                            samples_dir, gt_dir,
                            width=128, height=128,
                            resize=False, normalized=True):
    """
    Load images from a given directory path

    Args:
      :param folder_path: path of source directory
      :param samples_dir: samples directory name
      :param gt_dir: groundtruth directory name
      :param width: width of image
      :param height: height of image
      :param resize: resize image
      :param normalized: normalize images

    Returns:
      :return dataset_sp: samples images
      :return dataset_gt: groudtruth images
    """
    dataset_sp = []
    dataset_gt = []

    sp_path = os.path.join(folder_path, samples_dir)
    gt_path = os.path.join(folder_path, gt_dir)

    # List of all input images path and their groundtruths
    sp_plist = sorted([os.path.join(sp_path, fname)
                       for fname in os.listdir(sp_path)
                       if (fname.endswith('.jpg') or
                           fname.endswith('.JPG') or
                           fname.endswith('.png'))])

    gt_plist = sorted([os.path.join(gt_path, fname)
                       for fname in os.listdir(gt_path)
                       if (fname.endswith('.jpg') or
                           fname.endswith('.JPG') or
                           fname.endswith('.png'))])

    # Loading images to dataset from the images path list
    print('---- Loading Samples Images ---- ')
    for img_path in sp_plist:
        image_hazy = load_image(img_path, width, height, resize)
        dataset_sp.append(image_hazy)
        print(f'{img_path}')

    print('---- Loading GroundTruth Images ---- ')
    for gt_path in gt_plist:
        image_gt = load_image(gt_path, width, height, resize)
        dataset_gt.append(image_gt)
        print(f'{gt_path}')

    dataset_sp = np.array(dataset_sp)
    dataset_gt = np.array(dataset_gt)

    print("Normalizing")
    if normalized:
        dataset_sp = dataset_sp / 255.0
        dataset_gt = dataset_gt / 255.0

    print('\n\n>>>> Summary <<<<')
    print(f'  - Samples path: {sp_path}')
    print(f'  - Number of samples: {len(dataset_sp)} {dataset_sp.shape}')
    print(f'  - GroundTruth path: {gt_path}')
    print(f'  - Number of GroundTruth: {len(dataset_gt)} {dataset_gt.shape}')

    print("Done")

    return dataset_sp, dataset_gt


# Load batch of images from folder
def load_batch_images_from_folder(folder_path, width=128, height=128,
                                  resize=False, normalized=False, grayscale=False):
    """
    Load images from a given directory path

    Args:
      :param folder_path: path of source directory
      :param width: width of image
      :param height: height of image
      :param resize: resize image
      :param normalized: normalize images
      :param grayscale: If True, returns grayscale images. Otherwise,
                        returns colored images.

    Returns:
      :return batch_images: samples images

    """
    batch_img = []

    # List of all input images path and their groundtruths
    bt_plist = sorted([os.path.join(folder_path, fname)
                       for fname in os.listdir(folder_path)
                       if (fname.endswith('.jpg') or
                           fname.endswith('.JPG') or
                           fname.endswith('.png'))])

    # Loading images to dataset from the images path list
    print('---- Loading Batch of Images ---- ')
    for img_path in bt_plist:
        image_hazy = load_image(img_path, width, height, resize, grayscale, normalized)
        batch_img.append(image_hazy)
        print(f'{img_path}')

    batch_img = np.array(batch_img)

    print('\n\n>>>> Summary <<<<')
    print(f'  - Samples path: {folder_path}')
    print(f'  - Number of samples: {len(batch_img)} {batch_img.shape}')

    print("Done")

    return batch_img


def save_predicted_images(images, path_out, name_out, size=4):
    """
    Saves the images resulting from a model's prediction.

    Args:
        images: 4D or 3D images array NumPy (B, H, W, 3) or (B, H, W).
        path_out: The directory where the images will be saved.
        name_out: The file name of the images.
    """
    for i, imagem in enumerate(images):
        try:
            plt.imsave(f"{path_out}/{fill_with_leading_zeros(i, size)}_{name_out}.png", imagem)
            print(f"{path_out}/{fill_with_leading_zeros(i, size)}_{name_out}.png saved")
        except Exception as e:
            print(f"Error saving image: {e}")


def fill_with_leading_zeros(number, size):
    """
    Formata um número para uma string com zeros à esquerda.

    Args:
      number (int or float or str): The numeric input value.
      size (int): The desired total length of the output string.

    Returns:
      str: The string formatted with leading zeros.
    """

    str_num = str(number)

    return str_num.zfill(size)
