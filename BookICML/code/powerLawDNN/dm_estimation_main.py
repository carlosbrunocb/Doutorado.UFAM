from depth_map.pl_pytorch_model import *
from depth_map.pl_depth_maps import *
from image_handling.pl_image_handling import *
from math_transformation.pl_transformation import *


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
    drive_path = 'dataset/apl/res/512x512'
    sp_dir = 'hazy'
    gt_dir = 'hazy_gt'
    dim = (512, 512)
    version = 'v2'

    sp_datase = load_batch_images_from_folder(drive_path,
                                              width=dim[0], height=dim[1],
                                              resize=True,
                                              normalized=True)

    ''' ----- Loading Pytorch Model ----- '''
    print("\n:::::::: Loading Depth Anything model ::::::::")
    device, processor, model = load_depth_map_anything(version)
    print("Model loaded successfully!\n")

    print("\n:::::::: Generating Depth Maps ::::::::")
    print(f":images_batch sp_datase = {sp_datase.shape}")
    batch_depth_map = generate_depth_maps(sp_datase, device, processor, model)

    # dir_name = 'dataset/apl/dm/v1'
    dir_name = 'dataset/apl/dm/v2'
    file_name_dm = 'dm_sots_outdoor'

    save_predicted_images(batch_depth_map, dir_name, file_name_dm)


if __name__ == '__main__':
    main()
