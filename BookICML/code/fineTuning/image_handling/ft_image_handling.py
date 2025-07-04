import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, width=128, height=128, resize=False):
    """
    Read image from path using tensorflow struct

    Args:
      image_path: path of image
      width: width of image
      height: height of image
      resize: resize image

    Returns:
      img: image
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    if resize:
        img = tf.image.resize_with_pad(img,
                                       height, width,
                                       method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, tf.uint8)

    return img


def load_images_from_folder(folder_path,
                            samples_dir, gt_dir,
                            width=128, height=128,
                            resize=False, normalized=True):
    """
    Load images from a given directory path

    Args:
      folder_path: path of source directory
      samples_dir: samples directory name
      gt_dir: groundtruth directory name
      width: width of image
      height: height of image
      resize: resize image
      normalized: normalize images

    Returns:
      dataset_sp: samples images
      dataset_gt: groudtruth images
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


def save_predicted_images(images, path_out, name_out):
    """
    Saves the images resulting from a model's prediction.

    Args:
        images: 4D images array NumPy.
        path_out: The directory where the images will be saved.
        name_out:
    """
    for i, imagem in enumerate(images):
        try:
            plt.imsave(f"{path_out}/{name_out}_{i}.png", imagem)
            print(f"{path_out}/{name_out}_{i}.png saved")
        except Exception as e:
            print(f"Error saving image: {e}")
