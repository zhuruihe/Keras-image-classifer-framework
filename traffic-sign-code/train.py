import matplotlib
matplotlib.use('Agg')

import argparse
import random
import cv2
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

sys.path.append('..')
from net.lenet import LeNet

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-dtest', '--dataset_test', required=True, help='Path to input dataset_test')
    ap.add_argument('-dtrain', '--dataset_train', required=True, help='Path to input dataset_train')
    ap.add_argument('-m', '--model', required=True, help='Path to output model')
    ap.add_argument('-p', '--plot', required=True, default='plot.png', help='Path to output accuracy/loss plot')
    args = vars(ap.parse_args())
    return args
    
EPOCHS = 35
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 62
norm_size = 32
    
def load_data(path):
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
        image = cv2.resize(image, (norm_size, norm_size))
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

def train(aug, trainX, trainY, testX, testY, args):
    # initialize the model
    print('[INFO] Compiling model...')
    model = LeNet.build(norm_size, norm_size, 3, CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # train the network
    print('[INFO] Training model...')
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), 
                            validation_data=(testX, testY), steps_per_epoch=len(trainX)//BS,
                            epochs=EPOCHS, verbose=1)
    
    # save the model to the disk
    print('[INFO] Serializing network...')
    model.save(args['model'])
    
    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy on traffic-sign classifier')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(args['plot'])
    
if __name__ == '__main__':
    args = args_parse()
    train_file_path = args['dataset_train']
    test_file_path = args['dataset_test']
    
    trainX, trainY = load_data(train_file_path)
    testX, testY = load_data(test_file_path)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, 
                             width_shift_range=0.1, height_shift_range=0.1, 
                             shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')
    train(aug, trainX, trainY, testX, testY, args)

