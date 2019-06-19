from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD

from  dog_cat_breeds.model import ModelPrep
from  dog_cat_breeds.dataloading import DataLoader

# read data
# build model
# train model + eval training
# eval on test data
# save model
# prepare for inference

NUM_CLASSES = 50    # TODO - modify according to real data
TRAIN_DATA_DIR = "data/train"
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 128

def training():

    model = ModelPrep.create_train_model(NUM_CLASSES, used_model=InceptionV3, optimizer='rmsprop')

    # train the model on the new data for a few epochs
    generator = DataLoader.get_image_generator(TRAIN_DATA_DIR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE,
                                               rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # train the model on the new data for a few epochs
    model.fit_generator(generator)

    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
    model = ModelPrep.prepare_model_for_fine_tune(model)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(generator)