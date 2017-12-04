#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:33:31 2017

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.arange(12).reshape(3,4)
A = A+1.0  
np.random.seed(1)
B = np.random.rand(3,4)


#print (np.linalg.inv(A)) Matrix ist nicht quadratisch

A = np.vstack ((A,[0,1,0,1]))
print (np.linalg.inv(A))

print( A*B )

print (B.T)

A = np.arange(12).reshape(3,4)
A = A+1.0 

print(A.dot(B.T))
print(B.dot(A.T))

SumB = B[0] + B[1] + B[2]
print (SumB)


MwB = (SumB[0] + SumB[1] + SumB[2] + SumB[3]) / (B.shape[0] * B.shape[1])
print(MwB)


print(np.matmul (A, B.T))
print(np.matmul (B, A.T))

B = np.sqrt(A)
print (B)

B=B*B

print (A)
print (B)

print (np.amin(B))

print (np.argwhere(B == np.max(B)))

print (np.flip(A[np.where(A%2 == 1)],0))

A = np.vstack((A,[13,14,15,16]))

v = np.ones((4,4))
b = np.ones((1,4))

print(A*v+b)

dv1 = np.array([0,0])
dv2 = np.array([4,4])

L1 = np.abs(dv1-dv2)
print (L1)

L2 = np.sqrt((dv1-dv2)**2) #FALSCH
print (L2)

print (np.linalg.norm((dv1-dv2), ord=1))    #L1 in np
print (np.linalg.norm((dv1-dv2), ord=2))    #L2 in npr 


"""Aufgabe 1_4 """

def myGrid (start, stop, samples):
    result = np.array([0,0])
    step = abs(start-stop)/(samples-1) #step ist korrekt
    for i in np.arange(start,(stop+step),step):
        result = np.vstack((result, [i, i+step]))
    #result = np.array([[i for i in np.arange(start, (stop),step)]for j in np.arange(start, (stop),step)])
    return result

xy = myGrid (0,1,5)
plt.scatter(xy[:0] , xy[:1])

xy = myGrid(-1,1,3)
print (xy)
