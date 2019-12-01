from typing import Optional

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
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt  # use as plt.imshow()

import os.path

from dataload import cats_processor, dogs_processor
from dataload.animal_processor_base import AnimalProcessorBase
from dataload.cats_processor import CatsProcessor
from dataload.dogs_processor import DogsProcessor
from model import ModelPrep
from dataloading import DataLoader, DataPrep

"""
The ML pipeline for classification of different animals breeds:
1. Recognize general animal (dog, cat, rabbit, bird, ...) and its bounding box
    a. For this, use pre-trained yolo v.3 model
2. For each animal class, train separate classifier based on existing model (Keras' Inception v3)
    -> train on existing datasets:
        a. Dogs: stanford_dogs (use annotations for getting cropped image and)
        b. Cats: cats from https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset
                 (annotated, not in classes for Generator)
"""

# read data
# build model
# train model + eval training
# eval on test data
# save model
# prepare for inference

IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 64
EPOCHS = 15

def get_train_generators():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generator for training images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_OUT, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, shuffle=True,
        class_mode='categorical')  # 'categorical' for multiple classes / 'binary' for 2 classes

    # Generator for validation images
    validation_generator = test_datagen.flow_from_directory(
        TEST_OUT, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, shuffle=False,
        class_mode='categorical')  # 'categorical' for multiple classes / 'binary' for 2 classes

    return train_generator, validation_generator


def get_callbacks():
    weights_file_path = os.path.join(MODEL_OUT, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint_callback = ModelCheckpoint(weights_file_path, save_best_only=True, save_weights_only=True)
    early_stop_callback = EarlyStopping(min_delta=0, patience=3, restore_best_weights=True)

    return checkpoint_callback, early_stop_callback


def load_weights(model: Model, w_path: str):
    model.load_weights(w_path)


def train_base(model: Model, num_classes: int):
    train_generator, validation_generator = get_train_generators()
    checkpoint_callback, early_stop_callback = get_callbacks()

    # first training
    print("First training of top layers ...")
    history = model.fit_generator(train_generator, steps_per_epoch=NUM_CLASSES * 200 / BATCH_SIZE,
                                  epochs=EPOCHS, validation_data=validation_generator,
                                  validation_steps=NUM_CLASSES * 50 / BATCH_SIZE,
                                  callbacks=[checkpoint_callback, early_stop_callback])


def train_finetune(model):
    # TODO: plot history
    train_generator, validation_generator = get_train_generators()
    checkpoint_callback, early_stop_callback = get_callbacks()

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("Model fine-tuning ...")
    history = model.fit_generator(train_generator, steps_per_epoch=NUM_CLASSES * 200 / BATCH_SIZE,
                                  epochs=EPOCHS, validation_data=validation_generator,
                                  validation_steps=NUM_CLASSES * 50 / BATCH_SIZE,
                                  callbacks=[checkpoint_callback, early_stop_callback])
    print("DONE train")


def evaluation(model, class_names=None):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    eval_generator = test_datagen.flow_from_directory(TEST_OUT, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical')

    Y_pred = model.predict_generator(eval_generator, NUM_CLASSES * 100 // BATCH_SIZE + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    conf_mat = confusion_matrix(eval_generator.classes, y_pred)
    print(conf_mat)
    print('Classification Report')
    labels = None
    if class_names:
        labels = class_names.keys()
    print(classification_report(eval_generator.classes, y_pred, target_names=labels))


def predict_one(file_name, model, class_names, im_size=(IMG_HEIGHT, IMG_WIDTH)):
    img_X = DataLoader.get_one_image(file_name, im_size=im_size)
    img_X = np.expand_dims(img_X, axis=0)  # add dimension for batch size
    y = model.predict(img_X, batch_size=1)

    max_class_idx = y.argmax()
    class_name = list(class_names.keys())[list(class_names.values()).index(max_class_idx)]
    print(class_name)
    return y, class_name


if __name__ == "__main__":

    # ---------------------  VARIABLE DEFINITIONS
    animal = "cat"
    # DATA_DIR = "c:\\wspace_other\\keras_tests\\data\\dogs-cats"
    DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"
    IMG_DIR = os.path.join(DATA_DIR, "images")
    TRAIN_OUT = os.path.join(DATA_DIR, "train")
    TEST_OUT = os.path.join(DATA_DIR, "test")
    INFERE_DIR = os.path.join(DATA_DIR, "infer")
    MODEL_OUT = os.path.join(DATA_DIR, "models")

    MODEL_FILE = os.path.join(MODEL_OUT, f"model_{animal}"+".{epoch:02d}-{val_loss:.2f}.hdf5")
    model_weights_file = os.path.join(MODEL_OUT, f"fine_model_{animal}"+".06-0.28.hdf5")

    # -----------------------  CODE FOR TRAINING / INFERENCE
    phase = "TRAIN"
    data_processor: Optional[AnimalProcessorBase] = None
    if animal == "cat":
        data_processor = CatsProcessor(DATA_DIR)
    elif animal == "dogs":
        data_processor = DogsProcessor(DATA_DIR)

    if data_processor is None:
        print("Unknown animal type !")
        exit(-1)

    if phase == "TRAIN":
        if data_processor.check_folder_structure():
            data_processor.create_folders_for_processing()

        NUM_CLASSES = data_processor.get_number_of_classes()

    was_fine_tuning = False
    num_classes = NUM_CLASSES

    train_generator, validation_generator = get_train_generators()
    class_names = (train_generator.class_indices)


    print("Creating first model...")
    model = ModelPrep.create_train_model(num_classes, used_model=Xception,
                                         optimizer='rmsprop', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    if phase == "TRAIN":
        if not was_fine_tuning:
            if model_weights_file:
                print("loading weights")
                load_weights(model, model_weights_file)
            train_base(model)

    print("Preparing model for fine tuning ...")
    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
    model = ModelPrep.prepare_model_for_fine_tune(model, metrics=["accuracy"])

    if model_weights_file:
        print("loading weights")
        load_weights(model, model_weights_file)

    if phase == "TRAIN":
        train_finetune(model)
    elif phase == "EVAL":
        evaluation(model, class_names)
    elif phase == "INFERE":
        img_to_predict = os.path.join(INFERE_DIR, "newfoundland1.jpg")
        predict_one(img_to_predict, model, class_names)
