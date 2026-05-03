# Importing Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Importing Dataset
data = pd.read_csv("/home/tehseen/Desktop/digit_recognizer_NN/mnist_train.csv")

# print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

print(m, n)

print(Y_dev.shape)
print(X_dev.shape)
