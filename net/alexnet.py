from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K

def build_alexnet(input_shape, nb_classes, dense_layers=2, 
                    hidden_units=4096, dropout_rate=0.5,
                    subsample_initial_block=False):
    model = Sequential()
    
    if subsample_initial_block:
        initial_kernel = (11,11)
        initial_strides= (4, 4 )
        model.add(Conv2D(96, initial_kernel, strides=initial_strides, input_shape=input_shape, 
              padding='valid', activation='relu', kernel_initializer='uniform'))
        # Initial subsample
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    else:
        initial_kernel = (11, 11)
        initial_strides= (1, 1)
        model.add(Conv2D(96, initial_kernel, strides=initial_strides, input_shape=input_shape, 
              padding='same', activation='relu', kernel_initializer='uniform'))
    
    
    model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    # subsample 1/2
    model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    # subsample 1/2
    model.add(Flatten())
    
    for i in range(dense_layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(nb_classes, activation='softmax'))
    return model

class AlexNet():
    @staticmethod
    def build(input_shape, nb_classes, dense_layers=2, 
              hidden_units=4096, dropout_rate=0.5, subsample_initial_block=False):
        model = build_alexnet(input_shape, nb_classes, dense_layers, hidden_units, dropout_rate, subsample_initial_block)
        return model