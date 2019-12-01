
import os
import shutil
import numpy as np
from six.moves import cPickle as pickle
from scipy import ndimage, io, misc

from xml.dom import minidom


IMG_SIZE = 300

class DataLoading_W_Annotation():
    """
    From https://github.com/saksham789/DOG-BREED-CLASSIFICATION-STANFORD-DOG-DATASET
    """    
    
    @staticmethod
    def create_structure(base_folder, images_folder):
        images_folders = os.listdir(images_folder)
        for folder in images_folders:
#             os.makedirs("train/"+folder.split("\\")[-1])
#             os.makedirs("test/"+folder.split("\\")[-1])
            os.makedirs(os.path.join(base_folder, 'cropped', 'train', folder.split("\\")[-1]))
            os.makedirs(os.path.join(base_folder, 'cropped', 'test', folder.split("\\")[-1]))
            
            
    @staticmethod
    def move_data_files(image_list, old_folder, new_folder):
        for file in image_list:
            old_name = os.path.join(old_folder, file[0][0])
            new_name = os.path.join(new_folder, file[0][0])
            if os.path.exists(old_name):
                shutil.move(old_name, new_name)
            elif not os.path.exists(new_name):
                print('%s does not exist, it may be missing' % new_name)
        return [new_folder+'/'+d for d in sorted(os.listdir(new_folder)) if os.path.isdir(os.path.join(new_folder, d))]
    
    
    @staticmethod
    def copy_cropped_files(image_list, old_folder, new_folder, annotation_folder):
        print(f'Processing list {image_list}')
        for file in image_list:
            old_name = os.path.join(old_folder, file[0][0])
            new_name = os.path.join(new_folder, file[0][0])
            if os.path.exists(old_name):                
                DataLoading_W_Annotation.save_cropped(file[0][0], old_folder, new_folder, annotation_folder)
            elif not os.path.exists(new_name):
                print('%s does not exist, it may be missing' % new_name)
    
    
    @staticmethod
    def save_cropped(file_name, old_folder, new_folder, annotation_folder, image_size=IMG_SIZE):
        old_name = os.path.join(old_folder, file_name)
        new_name = os.path.join(new_folder, file_name)
        annot_name = os.path.join(annotation_folder, file_name.split('.')[0])
        try:
            image_data = misc.imread(old_name)
            annon_xml = minidom.parse(annot_name)
            xmin = int(annon_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            ymin = int(annon_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            xmax = int(annon_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            ymax = int(annon_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            
            new_image_data = image_data[ymin:ymax,xmin:xmax,:]
            new_image_data = misc.imresize(new_image_data, (image_size, image_size))
            misc.imsave(new_name, new_image_data)
            print(f'...saving file {new_name}')
        except IOError as e:
            print('Could not read:', old_name, ':', e, '- it\'s ok, skipping.')
    
        
    @staticmethod
    def load_breed(folder, image_size, num_channels, annotation_folder):
        """
        Load the data for a single breed label.
        Also, save cropped image to 'cropped' folder 
        """
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), image_size, image_size,num_channels), dtype=np.float32)
        print(folder)
        num_images = 0
        for image in image_files:
            image_file = folder+'/'+image
            try:
                
                image_data = misc.imread(image_file)
                
                annon_file = os.path.join(annotation_folder, folder.split('/')[-1], image.split('.')[0])
                annon_xml = minidom.parse(annon_file)
                xmin = int(annon_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
                ymin = int(annon_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
                xmax = int(annon_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
                ymax = int(annon_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)
                
                new_image_data = image_data[ymin:ymax,xmin:xmax,:]
                new_image_data = misc.imresize(new_image_data, (image_size, image_size))
                misc.imsave('cropped/' + folder + '/' + image, new_image_data)
                dataset[num_images, :, :, :] = new_image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
        dataset = dataset[0:num_images, :, :, :]
    
        print('Full dataset tensor:', dataset.shape)
        return dataset
    
    
    @staticmethod
    def maybe_pickle(data_folders, force=False):
        """
        Load dataset array and pickle it for faster processing and memory saving
        @return: list of directories / breeds
        """
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = DataLoading_W_Annotation.load_breed(folder)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)
      
        return dataset_names
    
    
    @staticmethod
    def make_arrays(nb_rows, img_size, num_channels):
        if nb_rows:
            dataset = np.ndarray((nb_rows,img_size, img_size,num_channels), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels
    
    
    @staticmethod
    def merge_datasets(pickle_files, train_size, image_size, valid_size=0, even_size=True):
        """
        Load training/validation data from pickle files
        @return: valid_dataset, valid_labels, train_dataset, train_labels
            where labels correspond to index of a pickle file
        """
        num_classes = len(pickle_files)
        valid_dataset, valid_labels = DataLoading_W_Annotation.make_arrays(valid_size, image_size)
        train_dataset, train_labels = DataLoading_W_Annotation.make_arrays(train_size, image_size)
        vsize_per_class = valid_size // num_classes
        tsize_per_class = train_size // num_classes
        
        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class+tsize_per_class
        for label, pickle_file in enumerate(pickle_files):
            try:
                with open(pickle_file, 'rb') as f:
                    breed_set = pickle.load(f)
                    np.random.shuffle(breed_set)
                    
                if not even_size:
                    tsize_per_class,end_l = len(breed_set),len(breed_set)
                    end_t = start_t + tsize_per_class
                    
                if valid_dataset is not None:
                    valid_breed = breed_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_breed
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
    
                
                train_breed = breed_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_breed
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise
        
        return valid_dataset, valid_labels, train_dataset, train_labels
    
    
    @staticmethod
    def read_mat_files(train_mat, test_mat, data_folder, annotation_folder, create_data_set=False, train_size = 9600, valid_size = 2400, test_size = 8580):
        
        test_list = io.loadmat(test_mat)['file_list']
        train_list = io.loadmat(train_mat)['file_list']
        
        old_folder = os.path.join(data_folder, 'Images')

        test_folders = DataLoading_W_Annotation.copy_cropped_files(test_list, old_folder, 
                                                                os.path.join(data_folder, 'cropped','test'),
                                                                annotation_folder)
        train_folders = DataLoading_W_Annotation.copy_cropped_files(train_list, old_folder,
                                                                os.path.join(data_folder, 'cropped','train'),
                                                                annotation_folder)
        
        if create_data_set:
            DataLoading_W_Annotation.create_data_labels_datasets(train_folders, test_folders, 
                                                                 train_size, valid_size, test_size)
        
    
    @staticmethod
    def create_data_labels_datasets(train_folders, test_folders, train_size = 9600, valid_size = 2400, test_size = 8580):
        train_datasets = DataLoading_W_Annotation.maybe_pickle(train_folders, force=True)
        test_datasets = DataLoading_W_Annotation.maybe_pickle(test_folders, force=True)
        
        valid_dataset, valid_labels, train_dataset, train_labels = DataLoading_W_Annotation.merge_datasets(
          train_datasets, train_size, valid_size)
        _, _, test_dataset, test_labels = DataLoading_W_Annotation.merge_datasets(test_datasets, test_size, even_size=False)
        
        print('Training:', train_dataset.shape, train_labels.shape)
        print('Validation:', valid_dataset.shape, valid_labels.shape)
        print('Testing:', test_dataset.shape, test_labels.shape)
        
        np.save(open('train_dataset.npy','wb'), train_dataset)
        np.save(open('train_labels.npy','wb'), train_labels)
        np.save(open('valid_dataset.npy','wb'), valid_dataset)
        np.save(open('valid_labels.npy','wb'), valid_labels)
        
        np.save(open('test_dataset.npy','wb'), test_dataset)
        np.save(open('test_labels.npy','wb'), test_labels)
    
    
if __name__ == "__main__":
    data_folder = "e:/data/dogs_cats/stanford_dogs"
    images_folder = os.path.join(data_folder, "Images")
    annotation_folder = os.path.join(data_folder, "Annotation")
    train_list = os.path.join(data_folder, 'train_list.mat')
    test_list = os.path.join(data_folder, 'test_list.mat')
    
    DataLoading_W_Annotation.create_structure(data_folder, images_folder)
    DataLoading_W_Annotation.read_mat_files(train_list, test_list, data_folder, annotation_folder)
    