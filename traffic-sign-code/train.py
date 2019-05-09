import matplotlib
matplotlib.use('Agg')

import argparse
import random
import cv2
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

sys.path.append('..')
from net.lenet import LeNet
from net.alexnet import AlexNet
from net.vgg import Vgg11, Vgg16, Vgg16_imageNet
from net.googlenetv1 import GoogLeNetV1
from net.resnet import ResNet
from net.densenet import DenseNet

def load_data(path, input_size):
    print('[INFO] Loading images...')
    data = []
    labels = []
    
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (input_size, input_size))
        image = img_to_array(image)
        data.append(image)
        
        # extract the class label from the image path and update the label list
        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)
        
    # scale the raw  pixel intensities to the range [0, 1]
    data = np.array(data , dtype='float32') / 255.0
    labels = np.array(labels)
    
    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels

def train(aug, trainX, trainY, testX, testY, batch_size, epochs,
          model, model_name, optimizer):
    
    

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    tensorboard = TensorBoard(log_dir=save_path + 'logs/{}'.format(model_name))
    # train the network
    print('[INFO] Training model {} ...'.format(model_name))
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), 
                            validation_data=(testX, testY), steps_per_epoch=len(trainX)//BS,
                            epochs=epochs, verbose=1, callbacks=[tensorboard])
    
   
        
    print('[INFO] Serializing network...')
    save_name = '{}_{:.2f}_{}'.format(model_name, 
                                      sum(H.history['val_acc'][-10:])/10., 
                                      epochs)
    model.save(save_path + save_name + '.model')
    
    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.plot(np.arange(0, N), np.ones(N), 'r-.', label='ground_acc')
    plt.title(model_name)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper right')
    plt.savefig(save_path + save_name + '.png')
    

    
# EPOCHS = 35
# INIT_LR = 1e-3
# BS = 32
# CLASS_NUM = 62
# norm_size = 32
# OP = 'adam'

train_file_path = '../traffic-sign/train'
test_file_path = '../traffic-sign/test'

if __name__ == '__main__':  
    
    EPOCHS = 3
    INIT_LR = 1e-3
    BS = 32
    CLASS_NUM = 62
    input_size = 32
    optimizer = 'adam'
    
    trainX, trainY = load_data(train_file_path, input_size)
    testX, testY = load_data(test_file_path, input_size)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, 
                             width_shift_range=0.1, height_shift_range=0.1, 
                             shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')
    
    input_shape = (input_size, input_size, 3)
    
    # build time stramp log dir
    time_stamp = time.strftime("%H%M%S", time.localtime())
    save_path = './{}_{}/'.format(time.strftime("%Y%m%d", time.localtime()), input_size)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    optimizer_zoo = ['adam', 'sgd']
    op_to_run = ['sgd', 'adam']
    
    model_zoo = ['lenet', 'alexnet', 'vgg11', 'vgg16', 'vgg_imageNet', 'googlenetv1', 'resnet', 'densenet']
    model_to_train = ['densenet']
    
    for op in op_to_run:
        if op_to_run == optimizer_zoo[0]:
            optimizer = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
        else:
            optimizer = SGD(lr=INIT_LR)
            
        for model_name in model_to_train:

            if model_name == model_zoo[0]: # LeNet
                model = LeNet.build(input_shape, CLASS_NUM)
            elif model_name == model_zoo[1]: # AlexNet
                model = AlexNet.build(input_shape, CLASS_NUM, dense_layers=2, 
                                      hidden_units=512, dropout_rate=0.5, 
                                      subsample_initial_block=False)
            elif model_name == model_zoo[2]: # VGG11
                model = Vgg11.build(input_shape, CLASS_NUM, dense_layers=2, 
                                  hidden_units=512, dropout_rate=0.5, sc_ratio=4)
            elif model_name == model_zoo[3]: # VGG16
                model = Vgg16.build(input_shape, CLASS_NUM, dense_layers=2, 
                                  hidden_units=512, dropout_rate=0.5, sc_ratio=4)
            elif model_name == model_zoo[4]: # VGG16 with ImageNet pretrained weights
                model = Vgg16_imageNet.build(input_shape, CLASS_NUM, dense_layers=2,
                                         hidden_units=512, dropout_rate=0.5,
                                         weights='imagenet')
            elif model_name == model_zoo[5]: # GoogLeNet V1
                model = GoogLeNetV1.build(input_shape, CLASS_NUM, dense_layers=1, 
                                          hidden_units=512, 
                                          subsample_initial_block=False)
            elif model_name == model_zoo[6]: # ResNet V1
                model = ResNet.build(input_shape, CLASS_NUM)
            elif model_name == model_zoo[7]: # DenseNet V1
                model = DenseNet.build(input_shape, CLASS_NUM, dense_layers=2, 
                                       hidden_units=512, dropout_rate=0.5,
                                       subsample_initial_block=False)
            else:
                # default lenet
                model = LeNet.build(input_shape, CLASS_NUM)
            # Start training process
            model_name = '{}_{}_{}_{}_{}'.format(model_name, time_stamp, op, INIT_LR, EPOCHS)
            train(aug, trainX, trainY, testX, testY, BS, EPOCHS,
                  model, model_name, optimizer)
    
    
    
    
    