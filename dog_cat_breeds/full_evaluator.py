from typing import Dict, List, Union, Any, Tuple
import os.path

import cv2
from keras import Model

from breed_evaluator import CroppedImgModeler
from dataload.animal_processor_base import AnimalProcessorBase
from utils import AnimalType, LoadedImage, BreedName
import yolo3_one_file_to_detect_them_all as yolo


class BreedPredictionUtils:

    def __init__(self, breed_modeler: CroppedImgModeler, model: Model, cls_names: Dict[Union[str, bytes], Any]):
        self.breed_modeler = breed_modeler
        self.model = model
        self.cls_names = cls_names


class ClassificationResult:

    def __init__(self, animal: AnimalType, breed: str):
        self.breed = breed
        self.animal = animal


class FullEvaluator:

    animals = List[AnimalType]
    models: Dict[AnimalType, BreedPredictionUtils]
    yolo_model: Model
    img_size: int

    def __init__(self, img_size: int):
        self.models = {}
        self.img_size = img_size
        self.phase = "INFERE"

    def load_models(self, models_dir) -> Dict[AnimalType, BreedPredictionUtils]:
        """
        Load trained models from files in folder,
        files follow structure as f"model_{animalType}_"+"{epoch:02d}-{val_loss:.2f}.hdf5"
        :param models_dir: path to saved models

        :return map of processors for each animal type
        """
        models = {}
        for model_file in os.listdir(models_dir):
            if model_file == "yolo_model.h5":
                self.yolo_model = yolo.load_model(model_file)
            if not model_file.endswith("hdf5"):
                continue
            parts = model_file.split("_")
            a_type = parts[1]
            breed_proc = self.prepare_breed_processor(a_type, model_file)
            models[a_type] = breed_proc

        return models

    def classify(self, img_path: str) -> ClassificationResult:
        """
        Classify image - first find the animal in the image and then classify breed
        :param img_path: path to image to classify
        :return: classification result - class and image
        """
        image = cv2.imread(img_path)
        # find all animals in the image
        found_animals = self.find_animal(img_path, image)
        # pick the largest (closest) one
        picked_animal, bbox = self.select_best_animal(found_animals)
        breed = self.classify_breed(image, picked_animal, bbox)
        return ClassificationResult(picked_animal, breed)

    def find_animal(self, img_path: str, image: LoadedImage) -> Dict[AnimalType, List[yolo.BoundBox]]:
        """
        Find an animal on the image, get its class and location
        :param img_path path to image to process
        :param image: loaded image handle
        :return map of found animals and their bounding boxes
        """
        found_animals = yolo.classify_image(image=image, img_path=img_path, yolov3=self.yolo_model,
                                            animals_to_find=list(self.models.keys()))
        return found_animals

    def classify_breed(self, image: LoadedImage, animal: AnimalType, bbox: yolo.BoundBox) -> BreedName:
        """
        Get the breed name of the found animal
        :param image: loaded image
        :param animal: animal type to classify
        :param bbox: found bounding box
        :return: name of the breed of the found animal
        """
        # TODO: check image types
        predict_utils = self.models[animal]
        # get sub-image
        cropped_image = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax, :]
        # pass it to the breed classifier
        breed_name = predict_utils.breed_modeler.predict_one_loaded(cropped_image,
                                                                    predict_utils.model, predict_utils.cls_names)
        return breed_name

    def prepare_breed_processor(self, animal_type: AnimalType, model_file: str) -> BreedPredictionUtils:
        """
        Prepare breed classifier
        :param animal_type: Animal type
        :param model_file: file with model weights
        :return: Animal processor
        """
        breed_modeler = CroppedImgModeler(animal_type, self.img_size)
        processor = breed_modeler.prepare_data(animal_type, self.phase)
        model = breed_modeler.prepare_model(self.phase, processor, True, model_file)
        cls_names = breed_modeler.get_class_names()
        bpu = BreedPredictionUtils(breed_modeler, model, cls_names)
        return bpu

    @staticmethod
    def select_best_animal(found_animals: Dict[AnimalType, List[yolo.BoundBox]]) \
            -> Tuple[AnimalType, yolo.BoundBox]:
        """
        Pick the most probably meant animal to classify from all that were found
        :param found_animals: map of found animals and their bounding boxes
        :return: pair of the AnimalType and the corresponding bounding box
        """
        lb_size = 0.0
        largest_box = None
        lb_animal = ""
        for animal in found_animals.keys():
            for bbox in found_animals[animal]:
                bb_size = (bbox.xmax-bbox.xmin) * (bbox.ymax-bbox.ymin)
                if bb_size > lb_size:
                    lb_size = bb_size
                    largest_box = bbox
                    lb_animal = animal
        return lb_animal, largest_box


if __name__ == "__main__":
    DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"
    MODEL_DIR = os.path.join(DATA_DIR, "models")
    img_to_classify = 'dog.jpg'

    evaluator = FullEvaluator(img_size=AnimalProcessorBase.IMG_SIZE)
    evaluator.load_models(MODEL_DIR)
    result = evaluator.classify(img_to_classify)
    print(f"Found animal {result.animal} - breed: {result.breed}")
