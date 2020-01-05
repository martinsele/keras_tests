from typing import Tuple

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD


# load existing model with weights, without top layers
# add top layers for classification
# build model

class ModelPrep:

    @staticmethod
    def create_train_model(num_outputs, used_model=InceptionV3, optimizer='rmsprop',
                           input_shape: Tuple[int, int, int] = (300, 300, 3)) -> Model:
        """
        Create model for fine-tuning
        :param used_model: type of model to load, e.g. Keras' InceptionV3
        :param num_outputs: number of outputs
        :param optimizer: optimizer to use
        :param input_shape: shape of input image
        :return: model without top layer
        """
        # create the base pre-trained model
        base_model = used_model(weights='imagenet', include_top=False, input_shape=input_shape)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add dropout
        x = Dropout(0.5)(x)

        # and a logistic layer
        predictions = Dense(num_outputs, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=optimizer, metrics=["accuracy"], loss='categorical_crossentropy')
        return model

    @staticmethod
    def prepare_fine_tuned_model(model: Model, metrics=None) -> Model:
        """
        Prepare model for fine-tuning
        :param model: model to tune
        :param metrics: metrics to observe
        :return:
        """
        # we chose to train the top 2 inception blocks
        for layer in model.layers[:-2]:
            layer.trainable = False
        for layer in model.layers[-2:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=metrics)
        return model
