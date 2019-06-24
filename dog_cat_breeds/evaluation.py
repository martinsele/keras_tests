from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from imageio import imread

import matplotlib.pyplot as plt # use as plt.imshow()

import os.path

from  model import ModelPrep
from  dataloading import DataLoader, DataPrep

# read data
# build model
# train model + eval training
# eval on test data
# save model
# prepare for inference

NUM_CLASSES = 37    # TODO - modify according to real data
DATA_DIR = "c:\\wspace_other\\keras_tests\\data\\dogs-cats"
# DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"
IMG_DIR = os.path.join(DATA_DIR, "images")
INFO_DIR = os.path.join(DATA_DIR, "annotations") 
TRAIN_OUT = os.path.join(DATA_DIR, "train")
TEST_OUT = os.path.join(DATA_DIR, "test")
MODEL_OUT = os.path.join(DATA_DIR, "models")

TRAIN_SPEC = os.path.join(INFO_DIR, "trainval.txt")
TEST_SPEC = os.path.join(INFO_DIR, "test.txt")
MODEL_FILE = os.path.join(MODEL_OUT, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 64
EPOCHS = 15



def get_train_generators():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    # Generator for training images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_OUT, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='categorical') # 'categorical' for multiple classes / 'binary' for 2 classes
    
    # Generator for validation images
    validation_generator = test_datagen.flow_from_directory(
        TEST_OUT, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='categorical')    # 'categorical' for multiple classes / 'binary' for 2 classes
    
    return train_generator, validation_generator


def get_callbacks():
    weights_file_path = os.path.join(MODEL_OUT, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint_callback = ModelCheckpoint(weights_file_path, save_best_only=True, save_weights_only=True)
    early_stop_callback = EarlyStopping(min_delta=0, patience=3, restore_best_weights=True)
    
    return checkpoint_callback, early_stop_callback


def load_weights(model, w_path):
    model.load_weights(w_path)


def train_base(model, num_classes=NUM_CLASSES):
    train_generator, validation_generator = get_train_generators()
    checkpoint_callback, early_stop_callback = get_callbacks()
    
    # first training
    print("First training of top layers ...")
    history = model.fit_generator(train_generator, steps_per_epoch=NUM_CLASSES*100/BATCH_SIZE,
        epochs=EPOCHS, validation_data=validation_generator,
        validation_steps=800, callbacks=[checkpoint_callback, early_stop_callback])


def train_finetune(model):
    #TODO: plot history
    train_generator, validation_generator = get_train_generators()
    checkpoint_callback, early_stop_callback = get_callbacks()
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("Model fine-tuning ...")
    history = model.fit_generator(train_generator, steps_per_epoch=500,
        epochs=EPOCHS, validation_data=validation_generator,
        validation_steps=200, callbacks=[checkpoint_callback, early_stop_callback])
    print("DONE train")
    

def cntn_training(model, model_weights_file, fine_tune=False, num_classes=NUM_CLASSES):
    # load weights
    print("loading weights")
    load_weights(model, model_weights_file)

    
if __name__ == "__main__":
    # DataPrep.prepare_data(train_spec=TRAIN_SPEC, train_out=TRAIN_OUT, test_spec=TEST_SPEC, test_out=TEST_OUT, img_dir=IMG_DIR)
#     target_classes, avg_train, avg_test = DataLoader.get_data_info(TRAIN_OUT, TEST_OUT)
#     print("Classification for {} classes, {} train / {} test examples on average".format(len(target_classes), avg_train, avg_test))
#     train_base(len(target_classes))
    weights_to_continue = None
    was_fine_tuning=False

    print("Creating first model...")
    model = ModelPrep.create_train_model(num_classes, used_model=Xception,
                                         optimizer='rmsprop', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    if not was_fine_tuning:
        if weights_to_continue:
            print("loading weights")
            load_weights(model, model_weights_file)

        train_base(model)

    print("Preparing model for fine tuning ...")
    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
    model = ModelPrep.prepare_model_for_fine_tune(model)

    if weights_to_continue:
        print("loading weights")
        load_weights(model, model_weights_file)

    train_finetune(model)