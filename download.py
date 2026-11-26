import os.path
import shutil

import kagglehub

DATA_PATH = os.path.join('data', )



print('Downloading data...')
paultimothymooney_chest_xray_pneumonia_path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
print('Copying data to project directory...')
shutil.copytree(paultimothymooney_chest_xray_pneumonia_path, DATA_PATH, dirs_exist_ok=True)

print('Data source import complete.')