from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from scipy.ndimage import imread
from imageio import imread
from skimage.transform import resize

import os.path
from shutil import copy

import matplotlib.pyplot as plt # use as plt.imshow()

# Load data from folder
# Split to Train/Test
# Use ImageGenerator for training data modification

# prepare data augmentation configuration

class DataLoader:
    
    @staticmethod
    def get_data_info(train_folder, test_folder):
        '''
        Get information about dataset
        @return: targets - classification classes
        @return: avg. num of training examples
        @return: avg. num of testing examples
        '''
        targets = []
        avg_train = 0
        for f in os.listdir(train_folder):
            avg_train += len(os.listdir(os.path.join(train_folder, f)))
            targets.append(f)
        
        avg_test = 0
        for f in os.listdir(test_folder):
            avg_test += len(os.listdir(os.path.join(test_folder, f)))
            
        return targets, avg_train/len(targets), avg_test/len(targets)

    @staticmethod
    def get_one_image(img_path, im_size=(150, 150), rescale=1./255):
        img = imread(img_path)
        img = resize(img, output_shape=im_size)
        img = img*rescale
        return img
    

class DataPrep:
    
    @staticmethod
    def prepare_data(train_spec, train_out, test_spec, test_out, img_dir):
        """
        read training/testing data and save them to dir train/class_name 
        for ImageDataGenerator.flow_from_directory()
        """
        # prepare training
        DataPrep.create_structure(in_spec=train_spec, out_dir=train_out, img_dir=img_dir)
        # prepare testing
        DataPrep.create_structure(in_spec=test_spec, out_dir=test_out, img_dir=img_dir)
        
        
    @staticmethod
    def create_structure(in_spec, out_dir, img_dir):
        """
        Used to convert /cats_dogs_breed_keggle/ files to ImageDataGenerator structure
        according to cats_dogs_breed_keggle/annotations/train.txt
        """
        # read training data and save them to dir train/class_name for ImageDataGenerator.flow_from_directory()
        class_set = set()
        with open(in_spec, encoding="utf-8") as file:
            for line in file:
                idx = line.rfind('_')
                class_name = line[:idx]
                img_name = line.split()[0] + ".jpg"
                print(img_name)
                
                # create classes directories
                class_path = DataPrep.prepare_class_dir(out_dir, class_name)
                if class_path:
                    # copy train files
                    img_source = os.path.join(img_dir, img_name)
                    print(img_source)
                    copy(img_source, class_path)
                    
     
     
    @staticmethod
    def create_structure_from_non_divided(img_dir, out_dir, train_num=100):
        """
        Prepare data for ImageDataGenerator in case of data divided to classes but not to train/test
        @param img_dir: directory with images
        @param out_dir: direcotry, where to create train/test structure
        @param train_num: number of training examples in all classes
        """
        train_dirs = {}
        test_dirs = {}
        for cls in os.listdir(img_dir): # list all classes in dir
            i = 0
            train_files = []
            test_files = []
            for file in os.listdir(cls):
                if i < train_num:
                    train_files.append(file)
                else:
                    test_files.append(file)
                i += 1
            train_dirs[cls] = train_dirs
            test_dirs[cls] = test_dirs
        
        DataPrep.create_train_test_structure(out_dir, train_dirs, test_dirs)
     
     
    @staticmethod
    def create_train_test_structure(out_dir, train_dirs, test_dirs):
        """
        TODO: create structure
        """
        pass
                    
    @staticmethod
    def prepare_class_dir(root_folder, class_name): 
        class_path = os.path.join(root_folder, class_name)
        try:
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
        except OSError:
            print ("Creation of the directory %s failed" % class_path)
            return None
        else:
            print("Class {} created".format(class_path))
            return class_path
        
        
    