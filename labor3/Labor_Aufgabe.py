# -*- coding: utf-8 -*-
"""
DLM Labor: Backpropagation
WS17/18

Denny Britz. "Implementing a Neural Network from Scratch in Python – An Introduction"
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datagenerator
from tqdm import tqdm

nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters
learning_rate = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

#Create data
def create_data(num_samples):
    X, y = datagenerator.generate_data(num_samples)
    return X, y

#Feedfoward pass
def feedforward(X, model):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    
    #W.x+b
    z1 = X.dot(W1) + b1

    #Activation function
    a1 = np.tanh(z1)
    
    #W.a1 + b
    z2 = a1.dot(W2) + b2
    
    #Softmax
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    model['a1'] = a1
    
    return probs, model

#Backward pass
def backprop(X, y, probs, model):
    W1 = model['W1']
    W2 = model['W2']
    a1 = model['a1']
    
    delta3 = np.array(probs)
    delta3[range(len(X)), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    
    # Regularisierung
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    
    deltas = { 'dW1' : dW1, 'db1' : db1, 'dW2' : dW2, 'db2' : db2}
    
    return deltas

def parameter_update(model, deltas, l_r):
    
    learning_rate = l_r
    
    dW1 = deltas['dW1']    
    db1 = deltas['db1']
    dW2 = deltas['dW2']
    db2 = deltas['db2']
    
    model['W1'] += -learning_rate * dW1
    model['b1'] += -learning_rate * db1
    model['W2'] += -learning_rate * dW2
    model['b2'] += -learning_rate * db2
    
    return model

#Validate
def validate(X_val, y_val, model):
    y_pred = predict(X_val, model)
    accuracy = np.mean(np.array(y_pred == y_val, dtype=np.uint))
    print ('Validation accuracy: ', accuracy)
    return accuracy

#Predict
def predict(X, model):
    probs, _ =feedforward(X, model)
    y_pred = np.argmax(probs, axis=1)
    return y_pred

def get_grid(start=0.0, stop=1.0, samples=100):
    p = np.linspace(start, stop, samples)
    return np.array([(x, y) for x in p for y in p])

def plot_results(X, y, model):
    xmin, xmax = X.min(), X.max()
    meshgrid = get_grid(xmin, xmax)
    
    plt.figure(1)        
    plt.cla()
    grid_pred = predict(meshgrid, model)
    colors = ['red' if p==0 else 'blue' for p in grid_pred]
    plt.scatter(meshgrid[:,0], meshgrid[:,1], c=colors, alpha=0.1, edgecolors=None)
    colors = ['red' if y_i==0 else 'blue' for y_i in y]        
    plt.scatter(X[:,0], X[:,1], c=colors)


# Train Model
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of neurons in the hidden layer
# - epochs: Number of passes through the training data for gradient descent
def train_model(X, y, nn_hdim, epochs=200):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    # Gradient descent. For the complete training data...
    for i in tqdm(range(0, epochs)):

        # Forward propagation
        probs, model = feedforward(X, model)

        # Backpropagation
        deltas = backprop(X, y, probs, model)

        # Gradient descent parameter update
        model = parameter_update(model, deltas, learning_rate/len(X))
        
    #Accuracy
    y_pred = np.argmax(probs, axis=1)
    accuracy = np.mean(np.array(y_pred == y, dtype=np.uint))
    print ('\nTraining accuracy for epoch %d : %f ' %(i, accuracy))

    return model

def main():
    #Generate Data
    X, y = create_data(300)
    
    #Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    #Train model
    model = train_model(X_train, y_train, 2, 200)
    
    #Valdiate model
    validate(X_val, y_val, model)
    
    #Show results
    plot_results(X_val, y_val, model)
    

if __name__=="__main__":
    main()
