import os.path
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple, DefaultDict, Optional

import numpy
import operator

from core import utils
from dataload.animal_processor_base import AnimalProcessorBase

"""
Process cats dataset, taken from cats and dogs breeds dataset at 
https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset

The dataset images and corresponding annotations of cat (and dog) breeds in separate folders.
These two folders consists of images and annotations of 12 cat breeds (and 25 dog breeds), each of around 200 images.
`annotations/list.txt` contains information about existing files, their labels and whether it is a cat or dog.

The following methods should read process the files and create a file structure exploitable by 
keras.preprocessing.image.ImageDataGenerator, i.e.

```
data/
  cropped/
    train/
        cat1/
            cat001.jpg
            cat002.jpg
            ...
        cat2/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        cat1/
            cat001.jpg
            cat002.jpg
            ...
        cat2/
            cat001.jpg
            cat002.jpg
            ...
```   
"""

file_num = 0
LabelName = str


class CatsProcessor(AnimalProcessorBase):

    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        numpy.random.seed(utils.rand_seed)

    @staticmethod
    def read_process_list(in_spec) -> Tuple[DefaultDict[LabelName, List[str]], DefaultDict[LabelName, int]]:
        """
        Read annotations/list.txt and parse existing cat breed labels and their corresponding image names
        :param in_spec: file with list of existing breeds and corresponding image files
        :return
            dictionary where keys are the cat breed labels and values are lists of corresponding file names
            dictionary with nums of data per class
        """
        print("Reading images...")
        data_dict = defaultdict(list)
        data_sizes = defaultdict(int)
        with open(in_spec, encoding="utf-8") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                # Lines ~ Image CLASS-ID SPECIES BREED_ID
                # where SPECIES ~ 1:Cat 2:Dog / BREED_ID ~ 1-25:Cat 1:12:Dog
                info_parts = line.split()
                if len(info_parts) < 4:
                    raise Exception(f"Incorrectly formated data for line {line}")
                # skip dog breeds
                if info_parts[2] == '2':
                    continue
                idx = info_parts[0].rfind('_')
                class_name = line[:idx]
                img_name = info_parts[0] + ".jpg"
                data_dict[class_name].append(img_name)
                data_sizes[class_name] += 1
                if data_sizes[class_name] == 1:
                    print(img_name)
        print(f"Images parsed: cat breeds:{len(data_sizes.keys())}")
        return data_dict, data_sizes

    def create_folder_structure(self, data_map: Dict[LabelName, List[str]], data_sizes: DefaultDict[LabelName, int],
                                base_folder: str, image_folder: str, annot_folder: str,
                                train_pct: float = 0.7, val_pct: float = 0.2):
        """
        Create folder structure exploitable by ImageGenerator and fill it with cropped images
        :param data_map: dictionary containing data files per class
        :param data_sizes: dictionary containing sizes of data per class
        :param base_folder:
        :param image_folder:
        :param annot_folder:
        :param train_pct:
        :param val_pct:
        :return:
        """
        print("Creating folder structure...")
        self.create_structure(base_folder, data_map.keys())

        min_data_size = min(data_sizes.values())
        train_len = int(min_data_size * train_pct)
        val_len = int(min_data_size * val_pct)
        test_len = int(min_data_size * (1 - train_pct - val_pct))
        print(f"Train size: {train_len}, valid_size: {val_len}, test_size: {test_len}")
        for label, files in data_map.items():
            # split data to train/val/test
            rand_order = numpy.random.permutation(len(files))

            # crop and save images
            train_files = operator.itemgetter(*rand_order[:train_len])(files)
            valid_files = operator.itemgetter(*rand_order[train_len:train_len + val_len])(files)
            test_files = operator.itemgetter(*rand_order[train_len + val_len:])(files)
            self.copy_cropped_files(train_files, image_folder,
                                    os.path.join(base_folder, "cropped", "train", label), annot_folder)
            self.copy_cropped_files(valid_files, image_folder,
                                    os.path.join(base_folder, "cropped", "valid", label), annot_folder)
            self.copy_cropped_files(test_files, image_folder,
                                    os.path.join(base_folder, "cropped", "test", label), annot_folder)
            # for full validation:
            self.copy_cropped_files(test_files, image_folder,
                                    os.path.join(base_folder, "non-cropped-test", label), None)

    @staticmethod
    def create_structure(base_folder: str, labels: Iterable[LabelName]):
        """
        Create a folder structure exploitable by ImageGenerator
        :param base_folder:
        :param labels: list of all breed labels/names
        """
        with open(os.path.join(base_folder, 'cat_class_names.txt'), mode='wt') as classes_file:
            for label in labels:
                classes_file.write(label + "\n")
                os.makedirs(os.path.join(base_folder, 'cropped', 'train', label), exist_ok=True)
                os.makedirs(os.path.join(base_folder, 'cropped', 'valid', label), exist_ok=True)
                os.makedirs(os.path.join(base_folder, 'cropped', 'test', label), exist_ok=True)
                # for full validation:
                os.makedirs(os.path.join(base_folder, 'non-cropped-test', label), exist_ok=True)

    def copy_cropped_files(self, image_list: List, old_folder: str, new_folder: str, annot_folder: Optional[str]):
        """
        Crop and copy files in image_list to a new folder
        :param image_list:
        :param old_folder:
        :param new_folder:
        :param annot_folder:
        :return:
        """
        print(f'Processing list {image_list}')
        for file in image_list:
            old_name = os.path.join(old_folder, file)
            new_name = os.path.join(new_folder, file)
            if os.path.exists(old_name):
                self.save_cropped(file, old_folder, new_folder, annot_folder)
            elif not os.path.exists(new_name):
                print('%s does not exist, it may be missing' % new_name)

    def create_folders_for_processing(self):
        """
        Process dataset and prepare it for ImageGenerator processing
        """
        images_folder = os.path.join(self.data_folder, "images")
        annotation_folder = os.path.join(self.data_folder, "annotations", "xmls")

        data_dct, size_dct = self.read_process_list(os.path.join(self.data_folder, "annotations", "list.txt"))
        self.create_folder_structure(data_dct, size_dct, self.data_folder, images_folder, annotation_folder,
                                     train_pct=0.7, val_pct=0.2)


if __name__ == "__main__":
    data_dir = "e:/data/dogs_cats/cats_dogs_breed_keggle"
    processor = CatsProcessor(data_dir)
    if not processor.check_folder_structure():
        processor.create_folders_for_processing()
