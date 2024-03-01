from tqdm import tqdm
from skimage import io
from skimage.transform import resize
import os
import numpy as np
from PIL import Image
#root_dir = '/home/nvidia/workspace/bishoy/Tirocinio/riconoscimento-dell-orecchio/EarVN1.0/EarVN1.0 dataset/public'
def loader(root_dir, target_size=(80,   80)):
    print(f"Loading dataset from {root_dir}")
    root_dir ='/Users/bishoyzakhary/Downloads/Ape/Images2'
  #  parent_path_list = os.listdir(root_dir)
    parent_path_list = [os.path.join(root_dir, dir) for dir in os.listdir(root_dir) if
                        os.path.isdir(os.path.join(root_dir, dir))]
    image_list = []
    label_list = []
    for parent_path in tqdm(parent_path_list):
        parent_path = os.path.join(root_dir, parent_path)
        child_path_list = os.listdir(parent_path)
        for child_path in child_path_list:
            image_path = os.path.join(parent_path, child_path)
            # Open image using PIL and convert to RGB
            image = Image.open(image_path).convert("RGB")
            # Resize the image using PIL
            resized_image = image.resize(target_size)
            # Convert the resized image to numpy array
            image_array = np.array(resized_image)
            image_list.append(image_array)
            # The first   3 characters of the parent folder contain the label/class of the image
            label = int(os.path.basename(parent_path)[:3])
            label_list.append(label)
    
    # Stack images into a single NumPy array
    image_array = np.stack(image_list)

    label_array = np.array(label_list)

    print(f'Dataset Dimension: {image_array.shape}')
    print(f'Labels Dimension: {label_array.shape}')
    return image_array, label_array





def pad_or_crop_image(image, target_size):
    # Get dimensions of the input image
    height, width = image.shape[:2]
    target_width, target_height = target_size
    # Calculate padding or cropping dimensions
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    crop_width = max(0, width - target_width)
    crop_height = max(0, height - target_height)
    # Pad or crop the image
    if pad_width > 0 or pad_height > 0:
        # If padding is needed, pad the image with zeros
        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left
        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top
        padded_image = np.pad(image, ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right), (0, 0)), mode='constant')
        return padded_image
    elif crop_width > 0 or crop_height > 0:
        # If cropping is needed, crop the image from the center
        crop_start_y = crop_height // 2
        crop_start_x = crop_width // 2
        cropped_image = image[crop_start_y:crop_start_y+target_height, crop_start_x:crop_start_x+target_width]
        return cropped_image
    else:
        # If no padding or cropping is needed, return the original image
        return image
