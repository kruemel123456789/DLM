#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:34:54 2017

@author: root
"""
import numpy as np

SEED = 4645
np.random.seed(SEED)
import keras

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal

from sklearn.model_selection import train_test_split

num_classes = 10

# Hyperparameters
batch_size = 16
epochs = 20
lr = 1

def load_data(grayscale):
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # convert to grayscale
    if grayscale:
        X_train = rgb2gray(X_train)
        X_test = rgb2gray(X_test)
    
    # split in train val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return X_train, y_train, X_val, y_val, X_test, y_test 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def main():
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(grayscale=True)
    
    # weight initialization
    weight_init = RandomNormal(mean=0.0, stddev=0.05)
    
    # activation function
    activation_function = 'sigmoid'
    
    # Create model
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(units=64, kernel_initializer=weight_init, activation=activation_function))
    model.add(Dense(units=128, kernel_initializer=weight_init, activation=activation_function))
    model.add(Dense(units=256, kernel_initializer=weight_init, activation=activation_function))
    model.add(Dense(units=512, kernel_initializer=weight_init, activation=activation_function))
    model.add(Dense(units=num_classes, kernel_initializer=weight_init, activation='softmax'))
    
    # compile
    sgd = SGD(lr=lr)#, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,#'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    tensorboard = TensorBoard(log_dir='./logs')
    
    # train
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              callbacks=[tensorboard])
    
    # test
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss: {} \nTest accuracy: {}%".format(score[0], score[1]))
