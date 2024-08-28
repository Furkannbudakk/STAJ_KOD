import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

mask_list = os.listdir(MASK_DIR)
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

for mask_name in tqdm.tqdm(mask_list):

    mask_name_without_ex = mask_name.split('.')[0]

    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name_without_ex+'.jpeg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)
    #########################################
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)

    if image is None or mask is None:
        print(f"Resim veya maske okunamadı {mask_name_without_ex}")
        continue

    result_image = image.copy()
    result_image[mask == 1, :] = [0, 0, 255]
    opac_image = (image/2 + result_image/2).astype(np.uint8)
    #########################################
    cv2.imwrite(image_out_path, opac_image)
    #########################################  
    if VISUALIZE: 
    #########################################
        """""
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Masked Image')
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()
        """""
        #########################################