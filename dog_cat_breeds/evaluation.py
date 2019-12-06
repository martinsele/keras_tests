from typing import Optional, Tuple, Dict, Union, Any

from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, Callback
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import os.path

from dataload.animal_processor_base import AnimalProcessorBase
from dataload.cats_processor import CatsProcessor
from dataload.dogs_processor import DogsProcessor
from model import ModelPrep
from dataloading import DataLoader

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


class CroppedImgModeler:

    def __init__(self, img_size: int, batch_size: int, epochs: int, data_dir: str, optimalizator='rmsprop'):
        self.image_size = img_size
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.optimalizator = optimalizator
        self.data_dir = data_dir

        self.img_dir = os.path.join(data_dir, "cropped")
        self.eval_dir = os.path.join(self.img_dir, "test")
        self.infer_dir = os.path.join(data_dir, "infer")
        self.model_dir = os.path.join(data_dir, "models")
        self.test_dir = os.path.join(data_dir, "test")

    def get_train_generators(self) -> Tuple[DirectoryIterator, DirectoryIterator]:
        """
        Get image generators from directories
        :param data_dir: directory with the data
        :return: train and validation data generators
        """
        train_dir = os.path.join(self.data_dir, "train")
        valid_dir = os.path.join(self.data_dir, "valid")
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        # Generator for training images
        train_gen = train_datagen.flow_from_directory(train_dir, target_size=(self.image_size, self.image_size),
                                                      batch_size=self.batch_size, shuffle=True,
                                                      class_mode='categorical')
        # 'categorical' for multiple classes / 'binary' for 2 classes

        # Generator for validation images
        valid_gen = valid_datagen.flow_from_directory(valid_dir, target_size=(self.image_size, self.image_size),
                                                      batch_size=self.batch_size, shuffle=False,
                                                      class_mode='categorical')
        return train_gen, valid_gen

    def get_callbacks(self) -> Tuple[Callback, ...]:
        """
        Prepare training callbacks
        :return tuple of training callbacks
        """
        weights_file_path = os.path.join(self.model_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
        checkpoint_callback = ModelCheckpoint(weights_file_path, save_best_only=True, save_weights_only=True)
        early_stop_callback = EarlyStopping(min_delta=0, patience=3, restore_best_weights=True)
        return checkpoint_callback, early_stop_callback

    @staticmethod
    def load_weights(model_to_load: Model, w_path: str):
        """
        Load model weights from file
        :param model_to_load: prepared model
        :param w_path: file with weights
        """
        model_to_load.load_weights(w_path)

    def train_base(self, base_model: Model, num_classes: int) -> History:
        """
        Train base model
        :param base_model: model to train
        :param num_classes: number of decision classes
        :return: history
        """
        train_gen, validation_gen = self.get_train_generators()
        checkpoint_callback, early_stop_callback = self.get_callbacks()

        # first training
        print("First training of top layers ...")
        history = base_model.fit_generator(train_gen, steps_per_epoch=num_classes * 200 / self.batch_size,
                                           epochs=self.num_epochs, validation_data=validation_gen,
                                           validation_steps=num_classes * 50 / self.batch_size,
                                           callbacks=[checkpoint_callback, early_stop_callback])
        return history

    def train_finetune(self, fine_model: Model, num_classes: int) -> History:
        """
        Fine-tune the model
        :param fine_model:
        :param num_classes:
        :return: training history
        """
        # TODO: plot history
        train_gen, validation_gen = self.get_train_generators()
        checkpoint_callback, early_stop_callback = self.get_callbacks()

        # we train our model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers
        print("Model fine-tuning ...")
        history = fine_model.fit_generator(train_gen, steps_per_epoch=num_classes * 200 / self.batch_size,
                                           epochs=self.num_epochs, validation_data=validation_gen,
                                           validation_steps=num_classes * 50 / self.batch_size,
                                           callbacks=[checkpoint_callback, early_stop_callback])
        print("DONE train")
        return history

    def evaluation(self, trained_model: Model, data_dir: str, class_names: Dict[int, str] = None) -> np.ndarray:
        """
        Evaluate the trained model
        :param trained_model: trained model
        :param data_dir: folder with the data
        :param class_names: class names to assign
        :return:
        """
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        eval_generator = test_datagen.flow_from_directory(self.test_dir, target_size=(self.image_size, self.image_size),
                                                          batch_size=self.batch_size, shuffle=False,
                                                          class_mode='categorical')

        Y_pred = trained_model.predict_generator(eval_generator,
                                                 len(class_names) * 100 // self.batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        conf_mat = confusion_matrix(eval_generator.classes, y_pred)
        print(conf_mat)
        print('Classification Report')
        labels = None
        if class_names:
            labels = list(class_names.keys())
        print(classification_report(eval_generator.classes, y_pred, target_names=labels))
        return y_pred

    @staticmethod
    def predict_one(file_name: str, trained_model: Model, class_names: Dict[Union[bytes, str], Any],
                    im_size: Tuple[int, int] = (AnimalProcessorBase.IMG_SIZE, AnimalProcessorBase.IMG_SIZE)):
        """
        Classify particular file
        :param file_name: file to classify
        :param trained_model: classification model
        :param class_names: list of labels to use
        :param im_size: size of the image
        :return:
        """
        img_x = DataLoader.get_one_image(file_name, im_size=im_size)
        img_x = np.expand_dims(img_x, axis=0)  # add dimension for batch size
        y = trained_model.predict(img_x, batch_size=1)

        max_class_idx = y.argmax()
        class_name = list(class_names.keys())[list(class_names.values()).index(max_class_idx)]
        print(class_name)
        return y, class_name

    def model_data(self, phase: str):
        """
        Perform modeling given on the assigned actio
        :param data_dir: dataset folder root
        :param phase: on of TRAIN, EVAL, INFER
        """
        model_file = os.path.join(self.model_dir, f"model_{animal}" + ".{epoch:02d}-{val_loss:.2f}.hdf5")
        model_weights_file = os.path.join(self.model_dir, f"fine_model_{animal}" + ".06-0.28.hdf5")

        # -----------------------  CODE FOR TRAINING / INFERENCE
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

        num_of_classes = data_processor.get_number_of_classes()
        was_fine_tuning = False

        train_generator, validation_generator = self.get_train_generators()
        cls_names = train_generator.class_indices

        print("Creating first model...")
        model = ModelPrep.create_train_model(num_of_classes, used_model=Xception,
                                             optimizer='rmsprop', input_shape=(self.image_size, self.image_size, 3))
        if phase == "TRAIN":
            if not was_fine_tuning:
                if model_weights_file:
                    print("loading weights")
                    self.load_weights(model, model_weights_file)
                self.train_base(model, num_of_classes)

        print("Preparing model for fine tuning ...")
        # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
        model = ModelPrep.prepare_model_for_fine_tune(model, metrics=["accuracy"])

        if model_weights_file:
            print("loading weights")
            self.load_weights(model, model_weights_file)

        if phase == "TRAIN":
            self.train_finetune(model, num_of_classes)
        elif phase == "EVAL":
            self.evaluation(model, self.eval_dir, cls_names)
        elif phase == "INFERE":
            img_to_predict = os.path.join(self.infer_dir, "newfoundland1.jpg")
            self.predict_one(img_to_predict, model, cls_names)


if __name__ == "__main__":
    # ---------------------  VARIABLE DEFINITIONS
    animal = "cat"
    # DATA_DIR = "c:\\wspace_other\\keras_tests\\data\\dogs-cats"
    DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"

    BATCH_SIZE = 64
    EPOCHS = 15
    img_modeler = CroppedImgModeler(AnimalProcessorBase.IMG_SIZE,
                                    data_dir=DATA_DIR,
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS
                                    )
