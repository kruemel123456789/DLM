# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:07:52 2016

@author: bkraus
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def generate_data(num_samples):
    # Daten generieren
    np.random.seed(0)
    samples = int(num_samples/2)
    mean1, cov1, = [.3, .7], [[.1,.1],[.1,.2]] 
    x1 = np.random.multivariate_normal(mean1, cov1, samples) # data
    y1 = np.zeros(samples, dtype=np.int)                     # target
    mean2, cov2 = [.6, .1], [[.1,0],[0,.1]]
    x2 = np.random.multivariate_normal(mean2, cov2, samples)
    y2 = np.ones(samples, dtype=np.int)
    
    x = np.concatenate((x1, x2), axis=0) 
    y = np.concatenate((y1, y2))
    
    return x, y
    
def generate_data_circles(num_samples):
    #Daten generieren
    np.random.seed(0)
    samples = 1000
    x, y = make_circles(samples, factor=0.5)
    return x, y