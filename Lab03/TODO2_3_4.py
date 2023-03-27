import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris


# Załaduj jako pandas dataframe
iris = load_iris(as_frame=True)
print('Database iris keys: ', iris.keys())
pd.set_option('display.max_columns', None)
# Wyswietl informacje o zbiorze
print(iris.frame.describe())

# 3. podziel na train i test
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

# bincount - liczy wystapienia poszczegolnych klas tj. majac macierz [0 1 1 4] po operacji bincount mamy [1 2 0 1], bo '0' wystapilo raz, '1' wystapilo dwa razy
print('Stratify:')
print(f'count y_train: {np.bincount(y_train)}')
print(f'count y_test: {np.bincount(y_test)}')



