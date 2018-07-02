# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:02:02 2018

@author: max_r

Function for least squares regression.
linLeastSquaresuares is only for linear regression.
LeastSquares function is for quadratic least squares regression.

A * c = y
Need to solve
A^T * A * C = A^T * y
"""

import numpy as np
import matplotlib.pyplot as plt

def linLeastSquares(x,y):
    ## create A
    A = np.ones((len(x),2))
    for i in range(len(x)):
        A[i,1] = x[i]
    
    A_T = np.transpose(A)
    
    y = np.dot(A_T,y)
    B = np.dot(A_T,A)
    A = np.linalg.inv(B)
    return np.dot(y,A)

def LeastSquares(x,y,n):
    ## create A
    A = np.ones((len(x),n+1))
    for i in range(1,n+1):
        for j in range(len(x)):
            A[j,i] = np.power(x[j],i)
            
    A_T = np.transpose(A)
    
    y = np.dot(A_T,y)
    B = np.dot(A_T,A)
    A = np.linalg.inv(B)
    return np.dot(y,A)
    

# %% create pseudo data
x = [i+1 for i in range(10)]
f = np.poly1d([2,0,0])
distortion = [20-(-20) * np.random.random_sample() + (-20) for i in range(10)]

polyDisturbed = f(x)
for i in range(10):
    polyDisturbed[i] += distortion[i] 
# change type to numpy type
x = np.asarray(x, dtype=np.float32)
polyDisturbed = np.asarray(polyDisturbed, dtype=np.float32)
#%% plot result
c = LeastSquares(x,polyDisturbed,2)
poly = np.poly1d([c[2],c[1],c[0]])

plt.plot(x,poly(x), x,polyDisturbed,'.')
