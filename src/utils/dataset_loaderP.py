from __future__ import print_function, division
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def loader(root_dir):
    print(f"Loading dataset from {root_dir}")
    root_dir = '/Users/bishoyzakhary/Desktop/università/Triennale/2022-2023/III anno/Laurea/Tirocinio/orecchie/AEPI/AEPI-Automated-Ear-Pinna-Identification-master/public'
    parent_path_list = os.listdir(root_dir)
    image_list = []
    label_list = []
    minimum_width = 80
    minimum_height = 80

    

    parent_path_list = os.listdir(root_dir)
    image_list = []
    label_list = []
    minimum_width = 80
    minimun_height = 80
    for parent_path in tqdm(parent_path_list):
        parent_path = os.path.join(root_dir, parent_path)
        child_path_list = os.listdir(parent_path)
        for child_path in child_path_list:
            image_path = os.path.join(parent_path,child_path)
            image = io.imread(image_path)
            rescaled_image = resize(image,(minimum_width,minimun_height))
                                
            #The first 3 literals of the parent folder contains the label/class of image.
            label = os.path.basename(parent_path)[:3]
            try:
                label = int(label)
                image_list.append(rescaled_image)
                label_list.append(label)
            except:
                print('Exception')
                    
    
    image_array = np.array(image_list)
    label_array = np.array(label_list)

    print(f'Dataset Dimesion: {image_array.shape}')
    print(f'Labels Dimension: {label_array.shape}')
    return (image_array , label_array)
