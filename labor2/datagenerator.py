# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:07:52 2016

@author: bkraus
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    # Daten generieren
    np.random.seed(0)
    samples = 150
    mean1, cov1, = [.3, .7], [[.1,.1],[.1,.2]] 
    x1 = np.random.multivariate_normal(mean1, cov1, samples) # data
    y1 = np.zeros(samples, dtype=np.int)                     # target
    mean2, cov2 = [.6, .1], [[.1,0],[0,.1]]
    x2 = np.random.multivariate_normal(mean2, cov2, samples)
    y2 = np.ones(samples, dtype=np.int)
    
    x = np.concatenate((x1, x2), axis=0) 
    y = np.concatenate((y1, y2))
    
    return x, y
    
    




