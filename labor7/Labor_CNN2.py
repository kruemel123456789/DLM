#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:03:58 2017

@author: root
"""


import numpy as np

SEED = 4645
np.random.seed(SEED)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal, glorot_uniform

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd

from Generator import Generator


FTRAIN = './data/training.csv'
FTEST = './data/test.csv'

# Hyperparameters
batch_size = 128
epochs = 20
lr = 1e-3
num_classes = 30


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    if y is not None:
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def plot_weights(weights):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    width, height, _, _ = weights.shape
    weights = weights.reshape(width, height, -1)   
    
    iters = np.minimum(weights.shape[-1], 16)
    for i in range(iters):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(weights[:, :, i], cmap='gray')
    plt.show()


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        #X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    return X, y


def main():
    
    # Load data
    X, y = load2d()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_test, _ = load2d(test=True)
    
    # Add data augmentation here!
    datagen_train = Generator(X_train, 
                              y_train, 
                              batchsize=batch_size,
                              flip_ratio=0.0) #!!!
    
    # weight initialization
    weight_init = glorot_uniform(seed=SEED)
    
    # activation function
    activation_function = 'relu'
    
    # Create model
    model = Sequential()
    model.add(Conv2D(filters=4,
                     kernel_size=3,
                     kernel_initializer=weight_init,
                     activation=activation_function,
                     padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=8,
                     kernel_size=2,
                     kernel_initializer=weight_init,
                     activation=activation_function,
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=8,
                     kernel_size=2,
                     kernel_initializer=weight_init,
                     activation=activation_function,
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=500,
                    kernel_initializer=weight_init,
                    activation=activation_function))
    model.add(Dense(units=num_classes,
                    kernel_initializer=weight_init,
                    activation='linear'))
    
    # compile
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True, decay=0.0)
    model.compile(optimizer=sgd,
                  loss='MSE')
    
    # Callbacks
    tensorboard = TensorBoard(log_dir='./logs/layeradded')
    
    # train (using our generator!)
    model.fit_generator(
        datagen_train.generate(),
        steps_per_epoch=(len(X_train)//batch_size),
        validation_data=(X_val, y_val),
        epochs=100,
        verbose=1,
        callbacks=[tensorboard])
    
    # test
    y_pred = model.predict(X_test, verbose=1)
    
    # Plot some prediction examples
    fig = plt.figure()
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X_test[i], y_pred[i], ax)
        
    # Plot some weights!
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            w, b = layer.get_weights()
            plot_weights(w)


if __name__ == "__main__":
    main()