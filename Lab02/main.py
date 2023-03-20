import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn import svm


# 1. klasyfikator SVC na bazie TODO2 z Lab01
# Wyswietlic po kilka elementow z kazdej z dostepnych klas
digits = datasets.load_digits()
# Convert 3D data to 2D data
data = digits.images.reshape(len(digits.images),-1)
X_train = []
y_train = []
for i in range(5):
    X_train.append(data[i])
    y_train.append(digits.target[i])
# print(digits.target[1])
svc = svm.SVC(gamma=0.001, C=100)
svc.fit(X_train, y_train)
X_test = np.array([X_train[1]])
y_test = np.array([y_train[1]])
y_pred = svc.predict(X_test)
print(y_pred)
print('Score: ', svc.score(X_test, y_test))

