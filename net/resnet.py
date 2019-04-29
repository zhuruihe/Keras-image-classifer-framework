from keras.models import Model, Sequential
from keras.layers import Input, concatenate, add
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D, ZeroPadding2D

def Conv2D_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Conv_Block(inputs, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
    x = Conv2D_BN(inputs, nb_filter=nb_filter[0], kernel_size=(1,1), strides=strides, padding='same')
    x = Conv2D_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3),  padding='same')
    x = Conv2D_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1),  padding='same')
    if with_conv_shortcut:
        shortcut = Conv2D_BN(inputs, nb_filter=nb_filter[2], kernel_size=kernel_size, strides=strides)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inputs])
        return x
    
def ResNet50(width, height, depth, classes):
    inputs = Input(shape=(width, height, depth))
    x = ZeroPadding2D((3,3), data_format='channels_last')(inputs)
    # 224, 224, 3
    x = Conv2D_BN(x, nb_filter=64, kernel_size=(7,7), strides=(2,2), padding='valid')
    # 112, 112, 64
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # 56, 56, 64
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3,3), strides=(1,1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    # 28, 28, 512
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    # 14, 14, 1024
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    # 7, 7, 2048
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3,3))
    
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

class ResNet():
    @staticmethod
    def build(width, height, depth, classes):
        model = ResNet50(width, height, depth, classes)
        return model