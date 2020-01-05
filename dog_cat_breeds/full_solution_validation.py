from typing import Dict

import utils
import os.path
from utils import AnimalType

cat = "cat"
dog = "dog"
# TODO: need non-cropped images for overall validation (same original files as the cropped ones)
valid_folders: Dict[AnimalType, str] = {cat: os.path.join(utils.DATA_DIRS[cat], "valid"),
                                        dog: os.path.join(utils.DATA_DIRS[dog], "valid")}

# TODO: list all valid_folders, extract files and their folder names (-> breed labels)
