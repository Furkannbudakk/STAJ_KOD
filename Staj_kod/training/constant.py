import os

JSON_DIR = '../SAT_Intern/data/jsons'
MASK_DIR  = '../SAT_Intern/data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)
IMAGE_OUT_DIR = '../SAT_INTERN/data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)
IMAGE_DIR = '../SAT_Intern/data/images'
VISUALIZE = True
BATCH_SIZE = 16
HEIGHT = 224
WIDTH = 224
N_CLASS= 2