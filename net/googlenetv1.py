from keras.models import Model, Sequential
from keras.layers import Input, concatenate, add
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D, ZeroPadding2D

def Conv2D_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding, name=conv_name, activation='relu')(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Inception(x, nb_filter):
    branch1x1 = Conv2D_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
    
    branch3x3 = Conv2D_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
    branch3x3 = Conv2D_BN(branch3x3, nb_filter, (3,3), padding='same', strides=(1,1), name=None)
    
    branch5x5 = Conv2D_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
    branch5x5 = Conv2D_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=None)
    
    branchpool= MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    branchpool= Conv2D_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
    
    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
    return x
    
def GoogLeNet(width, height, depth, classes):
    inputs = Input(shape=(width, height, depth))
    x = Conv2D_BN(inputs, 64, (7,7), strides=(2,2), padding='same')
    # 112, 112, 64
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # 56, 56, 64
    x = Conv2D_BN(x, 192, (3,3), strides=(1,1), padding='same')
    # 56, 56, 192
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # 28, 28, 192
    x = Inception(x, 64) # 256
    x = Inception(x, 120) # 480
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # 14, 14, 480
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)
    x = Inception(x, 208)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # 7, 7, 832
    x = Inception(x, 208)
    x = Inception(x, 256)
    x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x, name='Inception')
    return model

class GoogLeNetV1():
    @staticmethod
    def build(width, height, depth, classes):
        model = GoogLeNet(width, height, depth, classes)
#         model.summary()
        return model
