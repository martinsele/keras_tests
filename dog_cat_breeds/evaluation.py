from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from scipy.ndimage import imread
from scipy.io import loadmat
from scipy.misc import imresize

import os.path
import collections
from shutil import copyfile

from  dog_cat_breeds.model import ModelPrep
from  dog_cat_breeds.dataloading import DataLoader

# read data
# build model
# train model + eval training
# eval on test data
# save model
# prepare for inference

NUM_CLASSES = 50    # TODO - modify according to real data
DATA_DIR = "../../data/dogs-cats/images"
INFO_DIR = "../../data/dogs-cats/annotations/"
TRAIN_SPEC = os.path.join(INFO_DIR, "trainval.txt")
TEST_SPEC = os.path.join(INFO_DIR, "test.txt")
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 64
EPOCHS = 15


# read training data and save them to dir train/class_name for ImageDataGenerator.flow_from_directory()
train_dict = collections.defaultdict(list) # key: class_name, value: list
with open(TRAIN_SPEC, encoding="utf-8") as file:
    for line in file:
        pass


def train():
    DataLoader.load_imgs_dog_cat(DATA_DIR,im_size=(150, 150))

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    model = ModelPrep.create_train_model(NUM_CLASSES, used_model=Xception, optimizer='rmsprop')

    # train the model on the new data for a few epochs
    for e in range(EPOCHS):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / BATCH_SIZE:
                # we need to break the loop by hand because the generator loops indefinitely
                break


    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    #                     steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=EPOCHS)

    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
    model = ModelPrep.prepare_model_for_fine_tune(model)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(generator)