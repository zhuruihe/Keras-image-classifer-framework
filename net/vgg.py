from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K
from keras.applications.vgg16 import VGG16

def vgg_block(num_convs, num_channels, k_size):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, (k_size,k_size), strides=(1,1), 
                       padding='same', activation='relu', kernel_initializer='uniform'))
        
    blk.add( MaxPooling2D(pool_size=(2,2), strides=(2,2)) )
    return blk

class Vgg16():
    @staticmethod
    def build(input_shape, nb_classes, dense_layers=2, 
          hidden_units=4096, dropout_rate=0.5, sc_ratio=None):
        model = Sequential()
        
        model.add(Conv2D(64, (11,11), strides=(1,1),
                         padding='same', activation='relu', kernel_initializer='uniform', input_shape=input_shape))
        model.add(Conv2D(64, (11,11), strides=(1,1),
                         padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)) )
        
        conv_arch = ((2, 128, 5), (3, 256, 3), (3, 512, 3), (3, 512, 3))
        
        if sc_ratio:
            conv_arch = ((2, 128//sc_ratio, 5), (3, 256//sc_ratio, 3), (3, 512//sc_ratio, 3), (3, 512//sc_ratio, 3))
        
        for (num_convs, num_channels, k_size) in conv_arch:
            model.add( vgg_block(num_convs, num_channels, k_size) )
            
        model.add( Flatten())
        
        for i in range(dense_layers):
            model.add( Dense(hidden_units, activation='relu'))
            model.add( Dropout(dropout_rate))
        
        model.add( Dense(nb_classes, activation='softmax'))
        
        return model
    
class Vgg11():
    @staticmethod
    def build(input_shape, nb_classes, dense_layers=2, 
          hidden_units=4096, dropout_rate=0.5, sc_ratio=None):
        
        model = Sequential()
        
        conv_arch = ((1, 128, 5), (2, 256, 3), (2, 512, 3), (2, 512, 3)) 
        
        if sc_ratio:
            conv_arch = ((1, 128//sc_ratio, 5), (2, 256//sc_ratio, 3), (2, 512//sc_ratio, 3), (2, 512//sc_ratio, 3))

        model.add(Conv2D(64, (11, 11), strides=(1,1), input_shape=input_shape, 
                         padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)) )
        
        for (num_convs, num_channels, k_size) in conv_arch:
            model.add(vgg_block(num_convs, num_channels, k_size))
        
        model.add( Flatten())
        
        for i in range(dense_layers):
            model.add( Dense(hidden_units, activation='relu'))
            model.add( Dropout(dropout_rate))
        
        model.add( Dense(nb_classes, activation='softmax'))

        return model
    
class Vgg16_imageNet():
    @staticmethod
    def build(input_shape, nb_classes, dense_layers=2, 
              hidden_units=4096, dropout_rate=0.5, weights=None):
        
        base_model = VGG16(include_top=False, weights=weights, input_shape=input_shape)

        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

        for i in range(dense_layers):
            top_model.add(Dense(hidden_units, activation='relu'))
            top_model.add(Dropout(dropout_rate))

        top_model.add(Dense(nb_classes, activation='softmax'))
        # top_model.summary()

        full_model = Sequential()
        full_model.add(base_model)
        full_model.add(top_model)
        # full_model.summary()
        
        return full_model