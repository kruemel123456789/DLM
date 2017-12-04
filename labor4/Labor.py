# -*- coding: utf-8 -*-
"""
DLM Labor: Weight Init and Activation Functions
WS17/18

"""
import numpy as np

SEED = 76238
np.random.seed(SEED)

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.utils import to_categorical

import matplotlib
import matplotlib.pyplot as plt

def load_mnist():
    ''' Loads MNIST dataset using scikit-learn datasets library
        70000 examples of handwritten digits of size 28x28 pixels,
        labeled from 0 to 9.
        original data: yann.lecun.com/exdb/mnist
    '''
    from sklearn.datasets import fetch_mldata
    
    mnist = fetch_mldata('MNIST original', data_home='./mnist')
    
    # Rescale the data
    X, y = mnist.data / 255., mnist.target

    # one hot encoding
    y = to_categorical(y, 10)
    
    # use traditional train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_images(images):
    ''' Plot 10 random images from the dataset '''
    fig = plt.figure()
    
    # Randomly choose the 10 images
    idx = range(len(images))
    np.random.shuffle(idx)
    images = images[idx[0:10], :]    
    
    # Reshape images for plotting
    images = [np.reshape(image, (28,28))[:,3:25] for image in images]
    image = np.concatenate(images, axis=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def main():
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    
    # Visualize some images
    plot_images(X_train)
    
    # Weight init
    weigth_initializer = RandomNormal(mean=0.0, stddev=0.01, seed=SEED)
    activation = 'sigmoid'
    
    # create model
    model = Sequential()
    ## layer 1
    model.add(Dense(units= ,
                    input_shape= , 
                    kernel_initializer=weigth_initializer))
    model.add(Activation(activation))
    ## layer 2
    ...
    ## layer 3
    ...
    .
    .
    .
    ## layer n
    ## predictions layer
    ...
    
    optimizer = SGD(lr=0.01)
    
    # Compile model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    # Train model
    model.fit(...
              validation_data= ,
              batch_size=10,
              epochs=10, 
              shuffle=True)
    
    # Evaluate model
    score = model.evaluate(X_test, y_test)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

if __name__=="__main__":
    main()