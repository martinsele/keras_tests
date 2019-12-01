import os
from abc import ABC, abstractmethod


class AnimalProcessorBase(ABC):

    data_folder: str    # folder with original data, annotations, etc.
    IMG_SIZE: int = 300

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
            if "cropped" in folder_name:
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
                test_f = os.listdir(self.data_folder)[0]
                num_classes = len(os.listdir(test_f))
                return num_classes
        return -1

    def get_data_dir(self) -> str:
        """
        :return: original data directory path
        """
        return self.data_folder
