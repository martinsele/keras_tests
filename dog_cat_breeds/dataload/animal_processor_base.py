import os
from abc import ABC, abstractmethod
from typing import Optional

from scipy import misc
import imageio
from xml.dom import minidom

from utils import IMG_SIZE


class AnimalProcessorBase(ABC):

    data_folder: str    # folder with original data, annotations, etc.

    @abstractmethod
    def create_folders_for_processing(self):
        """
        Process dataset and prepare it for ImageGenerator processing
        """

    def check_folder_structure(self) -> bool:
        """
        Return True if dataset was already prepared for ImageGenerator processing
        :return: True / False
        """
        for folder_name in os.listdir(self.data_folder):
            if folder_name.startswith("cropped"):
                return True
        return False

    def get_number_of_classes(self) -> int:
        """
        :return:  Number of classification classes to be recognized
        """
        if not self.check_folder_structure():
            print("Data needs to be first processed into correct structure - use create_folders_for_processing()")
            return -1

        for folder_name in os.listdir(self.data_folder):
            if "cropped" in folder_name:
                test_f = os.listdir(os.path.join(self.data_folder, folder_name))[0]
                num_classes = len(os.listdir(os.path.join(self.data_folder, folder_name, test_f)))
                return num_classes
        return -1

    def get_data_dir(self) -> str:
        """
        :return: original data directory path
        """
        return self.data_folder

    # noinspection PyUnboundLocalVariable
    @staticmethod
    def save_cropped(file_name: str, old_folder: str, new_folder: str, annot_folder: Optional[str],
                     image_size: int = IMG_SIZE, annot_file_suffix: str = ".xml"):
        """
        Crop a file according to its corresponding annotation and save it to a new folder structure
        :param file_name:
        :param old_folder:
        :param new_folder:
        :param annot_folder:
        :param image_size:
        :param annot_file_suffix:
        """
        old_name = os.path.join(old_folder, file_name)
        new_name = os.path.join(new_folder, file_name)
        crop = False if annot_folder is None else True
        if crop:
            annot_name = os.path.join(annot_folder, file_name.split('.')[0]) + annot_file_suffix
        try:
            image_data = misc.imread(old_name)
            if crop and os.path.exists(annot_name):
                annon_xml = minidom.parse(annot_name)
                xmin = int(annon_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
                ymin = int(annon_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
                xmax = int(annon_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
                ymax = int(annon_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)
                new_image_data = image_data[ymin:ymax, xmin:xmax, :]
                new_image_data = misc.imresize(new_image_data, (image_size, image_size))
            else:
                new_image_data = image_data

            imageio.imsave(new_name, new_image_data[:, :, :3])
        except IOError as e:
            print('Could not read:', old_name, ':', e, '- it\'s ok, skipping.')

