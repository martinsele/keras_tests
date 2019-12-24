from typing import Dict, List, Union, Any
import os.path

from keras import Model

from breed_evaluator import CroppedImgModeler
from dataload.animal_processor_base import AnimalProcessorBase
from utils import AnimalType


class BreedPredictionUtils:

    def __init__(self, breed_modeler: CroppedImgModeler, model: Model, cls_names: Dict[Union[str, bytes], Any]):
        self.breed_modeler = breed_modeler
        self.model = model
        self.cls_names = cls_names


class BreedEvaluator:
    animals = List[AnimalType]
    models: Dict[AnimalType, BreedPredictionUtils]
    img_size: int

    def __init__(self, img_size: int, models_dir: str):
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
            if not model_file.endswith("hdf5"):
                continue
            parts = model_file.split("_")
            a_type = parts[1]
            breed_proc = self.prepare_breed_processor(a_type, model_file)
            models[a_type] = breed_proc
        return models

    def classify(self):
        pass

    def classify_type(self):
        pass

    def classify_breed(self):
        pass

    def prepare_breed_processor(self, type: AnimalType, model_file: str) -> BreedPredictionUtils:
        """
        Prepare breed classifier
        :param type: Animal type
        :param model_file: file with model weigths
        :return: Animal processor
        """
        breed_modeler = CroppedImgModeler(type, self.img_size)
        processor = breed_modeler.prepare_data(type,self.phase)
        model = breed_modeler.prepare_model(self.phase, processor, True, model_file)
        cls_names = breed_modeler.get_class_names()
        bpu = BreedPredictionUtils(breed_modeler, model, cls_names)
        return bpu


if __name__ == "__main__":
    DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"
    MODEL_DIR = os.path.join(DATA_DIR, "models")

    evaluator = BreedEvaluator(img_size=AnimalProcessorBase.IMG_SIZE, models_dir=MODEL_DIR)

    img_modeler.model_data(phase="TRAIN", fine_tune=fine_tune_model)
