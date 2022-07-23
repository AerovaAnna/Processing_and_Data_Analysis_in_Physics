
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from scipy import optimize
from sklearn.model_selection import train_test_split


Leaf = namedtuple('Leaf', ('value', 'x', 'y'))
Node = namedtuple('Node', ('feature', 'value', 'impurity', 'left', 'right',))

class BaseDecisionTree:
    def __init__(self, x, y, max_depth=np.inf):
        self.x = np.atleast_2d(x)
        self.y = np.atleast_1d(y)
        self.max_depth = max_depth
        
        self.features = x.shape[1]
        
        self.root = self.build_tree(self.x, self.y)
    
    # Will fail in case of depth ~ 1000 because of limit of recursion calls
    def build_tree(self, x, y, depth=1):
        if depth > self.max_depth or self.criteria(y) < 1e-6:
            return Leaf(self.leaf_value(y), x, y)
        
        feature, value, impurity = self.find_best_split(x, y)
        
        left_xy, right_xy = self.partition(x, y, feature, value)
        left = self.build_tree(*left_xy, depth=depth + 1)
        right = self.build_tree(*right_xy, depth=depth + 1)
        
        return Node(feature, value, impurity, left, right)
    
    def leaf_value(self, y):
        raise NotImplementedError
    
    def partition(self, x, y, feature, value):
        i = x[:, feature] >= value
        j = np.logical_not(i)
        return (x[j], y[j]), (x[i], y[i])
    
    def _impurity_partition(self, value, feature, x, y):
        (_, left), (_, right) = self.partition(x, y, feature, value)
        return self.impurity(left, right)
    
    def find_best_split(self, x, y):
        best_feature, best_value, best_impurity = 0, x[0,0], np.inf
        for feature in range(self.features):
            if x.shape[0] > 2:
                x_interval = np.sort(x[:,feature])
                res = optimize.minimize_scalar(
                    self._impurity_partition, 
                    args=(feature, x, y),
                    bounds=(x_interval[1], x_interval[-1]),
                    method='Bounded',
                )
                assert res.success
                value = res.x
                impurity = res.fun
            else:
                value = np.max(x[:,feature])
                impurity = self._impurity_partition(value, feature, x, y)
            if impurity < best_impurity:
                best_feature, best_value, best_impurity = feature, value, impurity
        return best_feature, best_value, best_impurity
    
    # Can be optimized for given .criteria()
    def impurity(self, left, right):
        h_l = self.criteria(left)
        h_r = self.criteria(right)
        return (left.size * h_l + right.size * h_r) / (left.size + right.size)
    
    def criteria(self, y):
        raise NotImplementedError
        
    def predict(self, x):
        x = np.atleast_2d(x)
        y = np.empty(x.shape[0], dtype=self.y.dtype)
        for i, row in enumerate(x):
            node = self.root
            while not isinstance(node, Leaf):
                if row[node.feature] >= node.value:
                    node = node.right
                else:
                    node = node.left
            y[i] = node.value
        return y

    
    

class DecisionTreeRegression(BaseDecisionTree):
    def __init__(self, x, y, *args, random_state=None, **kwargs):
        y = np.asarray(y)
        super().__init__(x, y, *args, **kwargs)
        
    def leaf_value(self, y):
        return np.mean(y)
    
    def criteria(self, y):
        p = np.std(y) 
        return p


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



data = pd.read_csv("sdss_redshift.csv")
x = data[["u", "g", "r", "i", "z"]]
y = data["redshift"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()


dtc = RandomForest(x_train, y_train, max_depth=3)
dtc.fit(x_train, y_train)

Y_train = dtc.predict_f(x_train)
Y_test = dtc.predict_f(x_test)

fig = plt.gcf()
plt.grid()
fig.set_size_inches(10,10)

plt.scatter(y_train, Y_train , color="y", label="train")
plt.scatter(y_test, Y_test , color="g",  label="test")
plt.xlabel('истинное значение')
plt.ylabel('предсказание')
plt.legend(loc='best')
fig.savefig('redhift.png')

file = {"train":  np.std(Y_train-y_train), "test": np.std(Y_test-y_test)}
with open('redhsift.json', 'w+') as f:
    json.dump(file, f)
    
    
data = pd.read_csv("C:\\Users\\Аня\\Desktop\\sdss.csv")
x = data[["u", "g", "r", "i", "z"]].to_numpy()
Y = dtc.predict_f(x)
dict = {"u": x[0:,0], "g":x[0:,1], "r":x[0:,2], "i":x[0:,3], "z":x[0:,4], "redshift": Y} 
df = pd.DataFrame(dict)
df.to_csv('sdss_predict.csv', index=False)
