from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K
from keras.applications.vgg16 import VGG16

class Vgg16():
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model.add( Conv2D(64, (3,3), strides=(1,1), input_shape=(width,height,depth), padding='same', activation='relu', kernel_initializer='uniform') )
        model.add( Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform') )
        model.add( MaxPooling2D(pool_size=(2,2)) )
        # 64, 112, 112
        model.add( Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 128, 56, 56
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 256, 28, 28
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 512, 14, 14
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 512, 7, 7
        model.add( Flatten())
        model.add( Dense(4096, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(4096, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(classes, activation='softmax'))

        return model
    
class Vgg11():
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model.add( Conv2D(64, (11,11), strides=(1,1), input_shape=(width,height,depth), padding='same', activation='relu', kernel_initializer='uniform') )
        model.add( MaxPooling2D(pool_size=(2,2)) )
        # 64, 112, 112
        model.add( Conv2D(128, (5,5), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 128, 56, 56
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 256, 28, 28
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 512, 14, 14
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add( MaxPooling2D(pool_size=(2,2)))
        # 512, 7, 7
        model.add( Flatten())
        model.add( Dense(1024, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(1024, activation='relu'))
        model.add( Dropout(0.5))
        model.add( Dense(classes, activation='softmax'))

        return model
    
class Vgg16_keras():
    @staticmethod
    def build(width, height, depth, classes):
        base_model = VGG16(include_top=False, weights=None, input_shape=(width, height, depth))
        # base_model.summary()

        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#         top_model.add(Dense(4096, activation='relu'))
#         top_model.add(Dropout(0.5))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(classes, activation='softmax'))
        # top_model.summary()

        full_model = Sequential()
        full_model.add(base_model)
        full_model.add(top_model)
        # full_model.summary()
        
        return full_model