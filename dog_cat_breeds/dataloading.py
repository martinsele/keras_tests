from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from scipy.ndimage import imread
from imageio import imread
from skimage.transform import resize

import os.path
from shutil import copy

import matplotlib.pyplot as plt  # use as plt.imshow()


# Load data from folder
# Split to Train/Test
# Use ImageGenerator for training data modification

# prepare data augmentation configuration

class DataLoader:

    @staticmethod
    def get_one_image(img_path, im_size=(150, 150), rescale=1. / 255):
        img = imread(img_path)
        img = resize(img, output_shape=im_size)
        img = img * rescale
        return img
