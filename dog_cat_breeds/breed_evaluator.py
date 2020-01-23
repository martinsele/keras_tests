from datetime import datetime
from typing import Optional, Tuple, Dict, Union, Any, List

from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, Callback, TensorBoard
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

import os.path

import utils
from dataload.animal_processor_base import AnimalProcessorBase
from dataload.cats_processor import CatsProcessor
from dataload.dogs_processor import DogsProcessor
from model import ModelPrep
from dataloading import DataLoader
from utils import AnimalType, BreedName

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

    def __init__(self, animal: AnimalType, img_size: int = utils.IMG_SIZE,
                 batch_size: int = 64, epochs: int = 10,
                 data_dir: str = "", optimizer='rmsprop',
                 short_experiment: bool = False):
        self.animal = animal
        self.image_size = img_size
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.optimizer = optimizer
        self.data_dir = data_dir
        self.short_experiment = short_experiment

        self.img_dir = os.path.join(data_dir, "cropped")
        self.eval_dir = os.path.join(self.img_dir, "test")
        self.infer_dir = os.path.join(data_dir, "infer")
        self.model_dir = os.path.join(data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def get_train_generators(self) -> Tuple[DirectoryIterator, DirectoryIterator]:
        """
        Get image generators from directories
        :return: train and validation data generators
        """
        train_dir = os.path.join(self.img_dir, "train")
        valid_dir = os.path.join(self.img_dir, "valid")
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

    def get_callbacks(self) -> List[Callback]:
        """
        Prepare training callbacks
        :return tuple of training callbacks
        """
        weights_file_path = os.path.join(self.model_dir, f"model_{self.animal}_"+"{epoch:02d}-{val_loss:.2f}.hdf5")
        checkpoint_callback = ModelCheckpoint(weights_file_path, save_best_only=True, save_weights_only=True)
        early_stop_callback = EarlyStopping(min_delta=0, patience=3, restore_best_weights=True)
        tensorboard = TensorBoard(log_dir='./log/{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        return [checkpoint_callback, early_stop_callback, tensorboard]

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
        callbacks = self.get_callbacks()

        # first training
        print("First training of top layers ...")

        steps_per_epoch = num_classes * 200 / self.batch_size
        validation_steps = num_classes * 50 / self.batch_size
        if self.short_experiment:
            steps_per_epoch = 10
            validation_steps = 10
        history = base_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                           epochs=self.num_epochs, validation_data=validation_gen,
                                           validation_steps=validation_steps,
                                           callbacks=callbacks)
        return history

    def train_finetune(self, fine_model: Model, num_classes: int) -> History:
        """
        Fine-tune the model
        :param fine_model:
        :param num_classes:
        :return: training history
        """
        train_gen, validation_gen = self.get_train_generators()
        callbacks = self.get_callbacks()

        # we train our model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers
        print("Model fine-tuning ...")
        steps_per_epoch = num_classes * 200 / self.batch_size
        validation_steps = num_classes * 50 / self.batch_size
        if self.short_experiment:
            steps_per_epoch = 10
            validation_steps = 10
        history = fine_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                           epochs=self.num_epochs, validation_data=validation_gen,
                                           validation_steps=validation_steps,
                                           callbacks=callbacks)
        print("DONE train")
        self.show_history(history, True)
        return history

    def evaluation(self, trained_model: Model, class_names: Dict[str, int] = None) -> np.ndarray:
        """
        Evaluate the trained model
        :param trained_model: trained model
        :param class_names: class names to assign
        :return:
        """
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        eval_generator = test_datagen.flow_from_directory(self.eval_dir, target_size=(self.image_size, self.image_size),
                                                          batch_size=self.batch_size, shuffle=False,
                                                          class_mode='categorical')

        y_pred_all = trained_model.predict_generator(eval_generator,
                                                     len(class_names) * 100 // self.batch_size + 1)
        y_pred = np.argmax(y_pred_all, axis=1)
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
                    im_size: Tuple[int, int] = (utils.IMG_SIZE, utils.IMG_SIZE)):
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

    @staticmethod
    def predict_one_loaded(image: np.ndarray, trained_model: Model,
                           class_names: Dict[Union[bytes, str], Any],
                           top_n: int = 3) -> Dict[BreedName, float]:
        """
        Classify a loaded and cropped image
        :param image: image to classify
        :param trained_model: classification model
        :param class_names: list of labels to use
        :param top_n: how many of best results to return
        :return:
            top N estimated BreedNames and their probability
        """
        top_results = {}
        img_x = np.expand_dims(image, axis=0)  # add dimension for batch size
        y = trained_model.predict(img_x, batch_size=1)

        max_class_idx = np.argpartition(y, -top_n)[0][-top_n:]
        for idx in max_class_idx:
            class_name = [cn for cn, i in class_names.items() if i == idx][0]
            top_results[class_name] = y[0, idx]
        return top_results

    def model_data(self, phase: str, fine_tune: bool, weights_to_load: str = ""):
        """
        Perform modeling given the assigned action
        :param phase: on of TRAIN, EVAL, INFER
        :param fine_tune: whether to fine-tune the model
        :param weights_to_load: path to weights file to load
        """
        # -----------------------  CODE FOR TRAINING / INFERENCE
        data_processor = self.prepare_data(self.animal, phase)
        num_of_classes = data_processor.get_number_of_classes()
        cls_names = self.get_class_names()

        model = self.prepare_model(phase, data_processor, fine_tune, weights_to_load)

        if phase == "TRAIN":
            self.train_finetune(model, num_of_classes)
        elif phase == "EVAL":
            self.evaluation(model, cls_names)
        elif phase == "INFERE":
            img_to_predict = os.path.join(self.infer_dir, "newfoundland1.jpg")
            self.predict_one(img_to_predict, model, cls_names)

    def get_class_names(self) -> Dict[str, int]:
        train_generator, _ = self.get_train_generators()
        cls_names = train_generator.class_indices
        return cls_names

    def prepare_data(self, animal: AnimalType, phase: str) -> AnimalProcessorBase:
        """
        Prepare data structure
        :param animal: animal data to process
        :param phase: on of TRAIN, EVAL, INFER
        :return prepared data processor
        """
        data_processor: Optional[AnimalProcessorBase] = None
        if animal == "cat":
            data_processor = CatsProcessor(self.data_dir)
        elif animal == "dog":
            data_processor = DogsProcessor(self.data_dir)

        if data_processor is None:
            print("Unknown animal type !")
            exit(-1)

        if phase == "TRAIN":
            if not data_processor.check_folder_structure():
                data_processor.create_folders_for_processing()
        return data_processor

    def prepare_model(self, phase: str, data_processor: AnimalProcessorBase,
                      fine_tune: bool, weights_to_load: str = "", num_classes: int = 0) -> Model:
        """
        Prepare model given the assigned action
        :param phase: on of TRAIN, EVAL, INFER
        :param data_processor: dataset processor
        :param fine_tune: whether to fine-tune the model
        :param weights_to_load: path to weights file to load
        :param num_classes: number of classes to predict

        :return prepared model with loaded weights
        """
        if num_classes == 0:
            num_classes = data_processor.get_number_of_classes()

        print("Creating first model...")
        model = ModelPrep.create_train_model(num_classes, used_model=Xception,
                                             optimizer=self.optimizer,
                                             input_shape=(self.image_size, self.image_size, 3))
        if phase == "TRAIN":
            if not fine_tune:
                if weights_to_load:
                    print("loading weights")
                    self.load_weights(model, weights_to_load)
                history = self.train_base(model, num_classes)
                self.show_history(history, False)

        print("Preparing model for fine tuning ...")
        # at this point, the top layers are well trained and we can start fine-tuning convolutional layers
        model = ModelPrep.prepare_fine_tuned_model(model, metrics=["accuracy"])
        if weights_to_load:
            print("loading fine-tuned weights")
            self.load_weights(model, weights_to_load)
        return model

    def show_history(self, history: History, is_finetuning: bool):
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model loss: finetuning {is_finetuning}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        plt.pause(0.1)


if __name__ == "__main__":
    # ---------------------  VARIABLE DEFINITIONS
    animal_type = "cat"
    fine_tune_model = False
    DATA_DIR = utils.DATA_DIRS[animal_type]
    # DATA_DIR = "c:\\wspace_other\\keras_tests\\data\\dogs-cats"

    BATCH_SIZE = 64
    EPOCHS = 20
    img_modeler = CroppedImgModeler(animal=animal_type, img_size=utils.IMG_SIZE, data_dir=DATA_DIR,
                                    batch_size=BATCH_SIZE, epochs=EPOCHS, short_experiment=False)

    model_weights_file_to_load = os.path.join(img_modeler.model_dir, f"fine-model_{animal_type}" + "_06-0.28.hdf5")
    img_modeler.model_data(phase="TRAIN", fine_tune=fine_tune_model)
    plt.show()
