from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K

class AlexNet():
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model.add( Conv2D(96, (11,11), strides=(4,4), input_shape=(width, height, depth), padding='valid', activation='relu', kernel_initializer='uniform'))
        # 96,55,55
        model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 96,27,27
        model.add( Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 256,13,13
        model.add( Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 256,6,6
        model.add( Flatten())
        model.add( Dense(4096, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(4096, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(classes, activation='softmax'))
        return model