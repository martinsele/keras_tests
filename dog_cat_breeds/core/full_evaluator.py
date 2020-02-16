from typing import Dict, List, Union, Any, Tuple
import os.path

import cv2
from keras import Model

from scipy import misc

from core.breed_evaluator import CroppedImgModeler
from core.utils import AnimalType, LoadedImage, BreedName, DATA_DIRS, NUM_CLASSES, CLASS_NAMES_FILES, IMG_SIZE, TOP_N
from core import yolo3_one_file_to_detect_them_all as yolo


class BreedPredictionUtils:

    def __init__(self, breed_modeler: CroppedImgModeler, model: Model, cls_names: Dict[Union[str, bytes], Any]):
        self.breed_modeler = breed_modeler
        self.model = model
        self.cls_names = cls_names


class ClassificationResult:

    def __init__(self, animal: AnimalType, breeds: Dict[BreedName, float]):
        self.breeds = breeds
        self.animal = animal

    def __repr__(self):
        return f"Animal: {self.animal} - breeds: {self.breeds}"


class FullEvaluator:

    animals = List[AnimalType]
    models: Dict[AnimalType, BreedPredictionUtils]
    yolo_model: Model
    img_size: int

    def __init__(self, img_size: int):
        self.models = {}
        self.img_size = img_size
        self.phase = "INFERE"

    def load_models(self, models_dirs: List[str]) -> Dict[AnimalType, BreedPredictionUtils]:
        """
        Load trained models from files in folder,
        files follow structure as f"model_{animalType}_"+"{epoch:02d}-{val_loss:.2f}.hdf5"
        :param models_dirs: paths to saved models

        :return map of processors for each animal type
        """
        models = {}
        for models_dir in models_dirs:
            for model_file in os.listdir(models_dir):
                if model_file == "yolo_model.h5":
                    self.yolo_model = yolo.load_model(os.path.join(models_dir, model_file))
                    continue
                if not model_file.endswith("hdf5"):
                    continue
                parts = model_file.split("_")
                a_type = str(parts[1])
                breed_proc = self.prepare_breed_processor(a_type, os.path.join(models_dir, model_file))
                models[a_type] = breed_proc

        self.models = models
        return models

    def classify(self, img_path: str, top_n: int = 3) -> ClassificationResult:
        """
        Classify image - first find the animal in the image and then classify breed
        :param img_path: path to image to classify
        :param top_n: number of top results to return
        :return: classification result - class and image
        """
        image = cv2.imread(img_path)

        # find all animals in the image
        found_animals = self.find_animal(img_path, image)
        if not found_animals:
            return ClassificationResult("nan", {"nan": 1.0})
        # pick the largest (closest) one
        picked_animal, bbox = self.select_best_animal(found_animals)

        breeds = self.classify_breed(image, picked_animal, bbox, top_n)
        return ClassificationResult(picked_animal, breeds)

    def find_animal(self, img_path: str, image: LoadedImage) -> Dict[AnimalType, List[yolo.BoundBox]]:
        """
        Find an animal on the image, get its class and location
        :param img_path path to image to process
        :param image: loaded image handle
        :return map of found animals and their bounding boxes
        """
        found_animals = yolo.classify_image(image=image, img_path=img_path, yolov3=self.yolo_model,
                                            animals_to_find=list(self.models.keys()), save_bbox=False)
        return found_animals

    def classify_breed(self, image: LoadedImage, animal: AnimalType, bbox: yolo.BoundBox, top_n: int) \
            -> Dict[BreedName, float]:
        """
        Get the breed name of the found animal
        :param image: loaded image
        :param animal: animal type to classify
        :param bbox: found bounding box
        :param top_n: number of top results to return
        :return: top N estimated BreedNames and their probability
        """
        predict_utils = self.models[animal]
        # get sub-image
        cropped_image = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax, :]
        new_image_data = misc.imresize(cropped_image, (IMG_SIZE, IMG_SIZE))
        # pass it to the breed classifier
        breed_names = predict_utils.breed_modeler.predict_one_loaded(new_image_data,
                                                                     predict_utils.model, predict_utils.cls_names,
                                                                     top_n)
        return breed_names

    def prepare_breed_processor(self, animal_type: AnimalType, model_file: str) -> BreedPredictionUtils:
        """
        Prepare breed classifier
        :param animal_type: Animal type
        :param model_file: file with model weights
        :return: Animal processor
        """
        breed_modeler = CroppedImgModeler(animal_type, self.img_size)
        processor = breed_modeler.prepare_data(animal_type, self.phase)
        model = breed_modeler.prepare_model(self.phase, processor, True, model_file,
                                            num_classes=NUM_CLASSES[animal_type])
        cls_names = self.get_class_names(animal_type)
        bpu = BreedPredictionUtils(breed_modeler, model, cls_names)
        return bpu

    @staticmethod
    def get_class_names(animal: AnimalType) -> Dict[BreedName, int]:
        """
        Read class names from file
        :param animal:  animal type
        :return: list of class names
        """
        num = 0
        class_names = {}
        file_name = CLASS_NAMES_FILES[animal]
        with open(file_name, mode='rt') as file:
            for line in file:
                class_names[line] = num
                num += 1
        return class_names

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
    model_dirs = []
    for data_dir in DATA_DIRS.values():
        model_dirs.append(os.path.join(data_dir, "models"))
    model_dirs.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models"))
    img_to_classify = os.path.join('..', 'dog.jpg')

    evaluator = FullEvaluator(img_size=IMG_SIZE)
    evaluator.load_models(model_dirs)
    result = evaluator.classify(img_to_classify, TOP_N)
    for res, prob in result.breeds.items():
        print(f"Found animal {result.animal} - breed: {res} ({prob*100:.2f})")
