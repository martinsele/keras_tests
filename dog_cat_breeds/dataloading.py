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
    
    @staticmethod
    def get_imgs_dog_cat(image_dir, im_size=(150, 150)):
        # load images
        basenames = [f for f in os.listdir(image_dir) if (f[-3:] == "jpg")][:]
        targs = ['_'.join(f.split("_")[:-1]) for f in basenames]
        image_paths = [os.path.join(image_dir, f) for f in basenames]

        # filter out black-white images
        images = [imread(imp) for imp in image_paths]
        grays = {i for i, im in enumerate(images) if (len(im.shape) != 3)}
        print("Total number of images: {}, color images: {}".format(len(images), len(grays)))

        images = [im for i, im in enumerate(images) if (i not in grays)]
        targs = [t for i, t in enumerate(targs) if (i not in grays)]
        images = [im if (im.shape[2] == 3) else im[:, :, :-1] for im in images]
        # resize all images
        images = [resize(im, size=im_size) for im in images]

        return image_paths, targs


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