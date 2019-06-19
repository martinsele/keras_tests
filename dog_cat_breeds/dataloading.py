from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load data from folder
# Split to Train/Test
# Use ImageGenerator for training data modification

# prepare data augmentation configuration

class DataLoader:

    @staticmethod
    def get_image_generator(data_dir, img_height, img_width, batch_size,
            rescale=1, shear_range=0, zoom_range=0, horizontal_flip=False):
        """
        Prepare data generator over directory
        TODO - change according to actual data structure
        :param data_dir: directory to generate images from
        :param img_height:
        :param img_width:
        :param batch_size:
        :param rescale:
        :param shear_range:
        :param zoom_range:
        :param horizontal_flip:
        :return:
        """

        train_datagen = ImageDataGenerator(rescale=rescale, shear_range=shear_range,
                                           zoom_range=zoom_range, horizontal_flip=horizontal_flip)

        train_generator = train_datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                            batch_size=batch_size, class_mode='binary')

        return train_generator
