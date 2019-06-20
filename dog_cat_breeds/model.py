from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD

# load existing model with weights, without top layers
# add top layers for classification
# build model

class ModelPrep:

    @staticmethod
    def create_train_model(num_outputs, used_model=InceptionV3, optimizer='rmsprop', input_shape=(299, 299, 3)):
        """
        Create model for fine-tuning
        :param used_model: e.g. Keras' InceptionV3
        :param num_outputs:
        :param optimizer: optimizer to use
        :return: model without top layer
        """
        # create the base pre-trained model
        base_model = used_model(weights='imagenet', include_top=False, input_shape=input_shape)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(num_outputs, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        return model


    @staticmethod
    def prepare_model_for_fine_tune(model):
        """
        Prepare model for fine-tuning
        :param model:
        :return:
        """
        # we chose to train the top 2 inception blocks
        for layer in model.layers[:-2]:
            layer.trainable = False
        for layer in model.layers[-2:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
        return model