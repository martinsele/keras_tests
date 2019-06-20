from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from scipy.ndimage import imread
from scipy.io import loadmat
from scipy.misc import imresize
import os

# Load data from folder
# Split to Train/Test
# Use ImageGenerator for training data modification

# prepare data augmentation configuration

class DataLoader:

    @staticmethod
    def get_imgs_dog_cat(image_dir, im_size=(150, 150)):
        # load images
        basenames = [f for f in os.listdir(image_dir) if (f[-3:] == "jpg")][:]
        targs = ['_'.join(f.split("_")[:-1]) for f in basenames]
        image_paths = [os.path.join(image_dir, f) for f in basenames]

        # filter out black-white images
        images = [imread(imp) for imp in image_paths]
        grays = {i for i, im in enumerate(images) if (len(im.shape) != 3)}
        print("Total number of images: {}, color images: {}".format(len(images), len(grays)))

        images = [im for i, im in enumerate(images) if (i not in grays)]
        targs = [t for i, t in enumerate(targs) if (i not in grays)]
        images = [im if (im.shape[2] == 3) else im[:, :, :-1] for im in images]
        # resize all images
        images = [imresize(im, size=im_size) for im in images]

        return image_paths, targs

    @staticmethod
    def get_image_generator(data_dir, img_height, img_width, batch_size,
            rescale=1, shear_range=0, zoom_range=0, horizontal_flip=False):
        """
        Prepare data generator over directory
        TODO - change according to actual data structure
        :param data_dir: directory to generate images from
        :param img_height:
        :param img_width:
        :param batch_size:
        :param rescale:
        :param shear_range:
        :param zoom_range:
        :param horizontal_flip:
        :return:
        """

        train_datagen = ImageDataGenerator(rescale=rescale, shear_range=shear_range,
                                           zoom_range=zoom_range, horizontal_flip=horizontal_flip)

        train_generator = train_datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                            batch_size=batch_size, class_mode='binary')

        return train_generator
