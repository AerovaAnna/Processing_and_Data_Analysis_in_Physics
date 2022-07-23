import matplotlib.pyplot as plt
import numpy as np
import  astropy.io.fits
from astropy.io import fits


data = fits.open('speckledata.fits')[2].data
#1
X = [X[1] for X in data ]
Y = [Y[2] for Y in data ]

for i in range(len(X)):
    X[i] = np.array(X[i]) 

A = []
B = []


for i in range(200):
    a = []
    for j in range(101):
        a.append(X[j][i])
    a = np.array(a)
    A.append(np.mean(a))
        
    
for i in range(200):
    a = []
    for j in range(101):
        a.append(Y[j][i])
    a = np.array(a)
    B.append(np.mean(a))
    
С = np.concatenate((A, B), axis=1)
img = Image.fromarray(С, 'RGB')
img.save('mean.png')
