import os.path
from typing import List, Optional

from scipy import io

from dataload.animal_processor_base import AnimalProcessorBase

"""
Process Stanford dogs dataset.
From https://github.com/saksham789/DOG-BREED-CLASSIFICATION-STANFORD-DOG-DATASET

The dataset Images and corresponding Annotations of dogs breeds in separate folders.
These two folders consists of 120 subfolders of various dog breeds, each of around 100 - 300 images.
`train_list.mat` and `test_list.mat` contains information about what files to use for training/testing.

The following methods should read process the files and create a file structure exploitable by 
keras.preprocessing.image.ImageDataGenerator, i.e.

```
data/
 cropped/
    train/
        dog1/
            dog001.jpg
            dog002.jpg
            ...
        dog2/
            dog001.jpg
            dog002.jpg
            ...
    validation/
        dog1/
            dog001.jpg
            dog002.jpg
            ...
        dog2/
            dog001.jpg
            dog002.jpg
            ...
```   
"""


class DogsProcessor(AnimalProcessorBase):

    def __init__(self, data_folder: str):
        self.data_folder = data_folder

    @staticmethod
    def create_structure(base_folder: str, img_folder: str):
        """
        Create a folder structure exploitable by ImageGenerator
        :param base_folder:
        :param img_folder:
        """
        with open(os.path.join(base_folder, 'dog_class_names.txt'), mode='wt') as classes_file:
            images_folders = os.listdir(img_folder)
            for folder in images_folders:
                c_name = folder.split("\\")[-1]
                classes_file.write(c_name + "\n")
                os.makedirs(os.path.join(base_folder, 'cropped', 'train', c_name), exist_ok=True)
                os.makedirs(os.path.join(base_folder, 'cropped', 'valid', c_name), exist_ok=True)
                os.makedirs(os.path.join(base_folder, 'cropped', 'test', c_name), exist_ok=True)
                # for full validation:
                os.makedirs(os.path.join(base_folder, 'non-cropped-test', c_name), exist_ok=True)

    def read_mat_files(self, train_mat: str, test_mat: str, base_folder: str, annot_folder: str):
        """
        Read .mat files and create train/test datasets within folder structure exploitable by ImageGenerator
        :param train_mat: mat file containing files to be used for training
        :param test_mat: mat file containing files to be used for testing
        :param base_folder: path to all dataset data
        :param annot_folder: path to annotations
        """
        tst_list = io.loadmat(test_mat)['file_list']  # 8580 images -> split to test/valid 50/50
        trn_list = io.loadmat(train_mat)['file_list']  # 12000 images

        old_folder = os.path.join(base_folder, 'Images')
        self.copy_cropped_files(tst_list[1::2], old_folder, os.path.join(base_folder, 'cropped', 'test'), annot_folder)
        self.copy_cropped_files(tst_list[0::2], old_folder, os.path.join(base_folder, 'cropped', 'valid'), annot_folder)
        self.copy_cropped_files(trn_list, old_folder, os.path.join(base_folder, 'cropped', 'train'), annot_folder)
        # for full validation:
        self.copy_cropped_files(tst_list[1::2], old_folder, os.path.join(base_folder, 'non-cropped-test'), None)

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
            old_name = os.path.join(old_folder, file[0][0])
            new_name = os.path.join(new_folder, file[0][0])
            if os.path.exists(old_name):
                self.save_cropped(file[0][0], old_folder, new_folder, annot_folder, annot_file_suffix="")
            elif not os.path.exists(new_name):
                print('%s does not exist, it may be missing' % new_name)

    def create_folders_for_processing(self):
        """
        Process dataset and prepare it for ImageGenerator processing
        """
        images_folder = os.path.join(self.data_folder, "Images")
        annotation_folder = os.path.join(self.data_folder, "Annotation")
        train_list = os.path.join(self.data_folder, 'train_list.mat')
        test_list = os.path.join(self.data_folder, 'test_list.mat')

        self.create_structure(self.data_folder, images_folder)
        self.read_mat_files(train_list, test_list, self.data_folder, annotation_folder)


if __name__ == "__main__":
    data_dir = "e:/data/dogs_cats/stanford_dogs"
    processor = DogsProcessor(data_dir)
    if processor.check_folder_structure():
        processor.create_folders_for_processing()
