
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



class RandomForest():
    def __init__(self, x, y, n=100, max_depth=3):
        self.x = np.atleast_2d(x)
        self.y = np.array(y)
        self.classes = np.unique(y)
        self.n = n
        self.max_depth = max_depth
        self.forest = []
        
    def fit(self, x, y):
        
        for i in range(self.n):
            rand_ind = np.unique(np.random.randint(0, self.x.shape[0], int(self.x.shape[0]/30)))
            self.forest.append(DecisionTreeRegression(x[rand_ind], y[rand_ind], max_depth=self.max_depth))
    
    def predict_f(self, x):
        x = np.atleast_2d(x)
        y = np.zeros(x.shape[0])
        for tree in self.forest:
            y += tree.predict(x)
        y = y / self.n 
        return np.array(y)
