import glob
import cv2
import torch
import numpy as np
from constant import *
import matplotlib.pyplot as plt


def tensorize_image(image_path_list, output_shape, cuda=False):
    local_image_list = []

    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image = cv2.resize(image, output_shape)
        torchlike_image = torchlike_data(image)
        local_image_list.append(torchlike_image)

    torch_image = torch.stack([torch.from_numpy(img).float() for img in local_image_list])

    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    local_mask_list = []

    for mask_path in mask_path_list:
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, output_shape)
        mask = one_hot_encoder(mask, n_class)
        if mask is not None:
            torchlike_mask = torchlike_data(mask)
            local_mask_list.append(torchlike_mask)

    if local_mask_list:
        torch_mask = torch.stack([torch.from_numpy(mask).float() for mask in local_mask_list])

        if cuda:
            torch_mask = torch_mask.cuda()

        return torch_mask
    else:
        print("No valid masks were processed")
        return None

def image_mask_check(image_path_list, mask_path_list):

    if len(image_path_list) != len(mask_path_list):
        return False
    
    image_names = {os.path.basename(path) for path in image_path_list}
    mask_names = {os.path.basename(path) for path in mask_path_list}

    return image_names == mask_names


############################ TODO ################################
def torchlike_data(data, is_mask=False):

    n_channels = data.shape[2] if len(data.shape) > 2 else 1
    torchlike_data_output = np.zeros((n_channels, data.shape[0], data.shape[1]))

    if n_channels == 1:
        torchlike_data_output[0] = data
    else:
        for channel in range(n_channels):
            torchlike_data_output[channel] = data[:, :, channel]

    return torchlike_data_output

def one_hot_encoder(data, n_class):

    if len(data.shape) != 2:
        print("It should be same with the layer dimension, in this case it is 2")
        return
    
    unique_values = np.unique(data)

    if len(unique_values) > n_class:

        threshold = np.max(unique_values) // 2
        binary_mask = (data > threshold).astype(np.int64)

        encoded_data = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.int64)
        encoded_data[:, :, 0] = 1 - binary_mask  # background
        encoded_data[:, :, 1] = binary_mask  # foreground

        return encoded_data

    encoded_data = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int64)

    for i, value in enumerate(unique_values):
        encoded_data[:, :, i] = (data == value).astype(int)

    return encoded_data
############################ TODO END ################################




if __name__ == '__main__':

    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()

    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()

    if image_mask_check(image_list, mask_list):

     batch_image_list = image_list[:BATCH_SIZE]
     batch_image_tensor = tensorize_image(batch_image_list, (224, 224))
 
     print("For features:\nType is " + str(batch_image_tensor.dtype))
     print("Type is " + str(type(batch_image_tensor)))
     print("The size should be [" + str(BATCH_SIZE) + ", 3, " + str(224) + ", " + str(224) + "]")
     print("Size is " + str(batch_image_tensor.shape) + "\n")

    
     batch_mask_list = mask_list[:BATCH_SIZE]
     batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT, WIDTH), 2)

     print(np.unique(batch_mask_tensor))
     print("For labels:\nType is " + str(batch_mask_tensor.dtype))
     print("Type is " + str(type(batch_mask_tensor)))
     print("The size should be [" + str(BATCH_SIZE) + ", 2, " + str(224) + ", " + str(224) + "]")
     print("Size is " + str(batch_mask_tensor.shape) + "\n") 

