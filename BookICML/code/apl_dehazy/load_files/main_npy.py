import numpy as np

from image_handling.apl_image_handling import save_predicted_images

# Construir o caminho completo do arquivo
file_path = '../results/dehaze/512x512/predictions_dm.npy'
dir_name = '../results/dehaze/512x512'
file_name_dm = 'dm_apl_sots_'


try:
    loaded_predicted_img = np.load(file_path)
    save_predicted_images(loaded_predicted_img, dir_name, file_name_dm)

except Exception as e:
    print(f"Error loading image: {e}")
