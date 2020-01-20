import os
from typing import Any

AnimalType = str   # animal: animal to classify ["cat", "dog", "nan"]
BreedName = str
LoadedImage = Any

rand_seed = 101

IMG_SIZE: int = 300
TOP_N = 3   # how many best results to return
CAT_DATA_DIR = "e:\\data\\dogs_cats\\cats_dogs_breed_keggle"
DOG_DATA_DIR = "e:\\data\\dogs_cats\\stanford_dogs"
DATA_DIRS = {"cat": CAT_DATA_DIR, "dog": DOG_DATA_DIR}
NUM_CLASSES = {"cat": 12, "dog": 120}

CLASS_NAMES_FILES = {"cat": os.path.join("models", "cat_class_names.txt"),
                     "dog": os.path.join("models", "dog_class_names.txt")}
