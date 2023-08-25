from __future__ import print_function , division
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np

def loader(root_dir):
    """ 
    Dataset Loader
    Example path: /home/user/EarVN1.0/Images 
    Args:
        root_dir(str): Root Directory of dataset folder
    Returns:
        (tuple): Image_array(np.array) , Label_array(np.array)
    """
    print(f"Loading dataset from {root_dir}")
    root_dir = 'EarVN1.0/EarVN1.0 dataset/Images'
    parent_path_list = os.listdir(root_dir)
    image_list = []
    label_list = []
    minimum_width = 80
    minimum_height = 80
    for parent_path in tqdm(parent_path_list):
        # Skip .DS_Store files
        if parent_path == '.DS_Store':
            continue
        parent_path = os.path.join(root_dir, parent_path)
        try:
            child_path_list = os.listdir(parent_path)
        except FileNotFoundError:
            print(f"Directory {parent_path} not found.")
            continue
        for child_path in child_path_list:
            image_path = os.path.join(parent_path, child_path)
            image = io.imread(image_path)
            rescaled_image = resize(image, (minimum_width, minimum_height))

            # The first 3 literals of the parent folder contain the label/class of the image.
            label = os.path.basename(parent_path)[:3]
            try:
                label = int(label)
                image_list.append(rescaled_image)
                label_list.append(label)
            except:
                print('Exception')

    image_array = np.array(image_list)
    label_array = np.array(label_list)

    print(f'Dataset Dimension: {image_array.shape}')
    print(f'Labels Dimension: {label_array.shape}')
    return (image_array, label_array)

