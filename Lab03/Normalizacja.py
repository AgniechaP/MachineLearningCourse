import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import make_classification
import pandas as pd
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Załaduj jako pandas dataframe
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

# Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
plt.scatter(X[:, 0], X[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.title('Iris sepal features')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

# Min-max scaler - ogranicz się do dwóch cech datasetu Iris
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train)
X_min_max_scaled = min_max_scaler.transform(X)

plt.scatter(X_min_max_scaled[:, 0], X_min_max_scaled[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.title('Iris sepal features - min max scaled')
plt.xlabel('sepal length (cm) - min max scaled')
plt.ylabel('sepal width (cm) - min max scaled')
plt.show()

# 6. Standaryzacja X' = (X - mi) / (sigma)
# mi - srednia, sigma - odchylenie standardowe

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.title('Iris sepal features - Standard scaled')
plt.xlabel('sepal length (cm) - standard scaled')
plt.ylabel('sepal width (cm) - standard scaled')
plt.show()

# 7. Pipelines i szeregowe łączenie modeli
# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline
pipe = Pipeline(
        [
            ('min_max_scaler', preprocessing.MinMaxScaler()),
            ('clf_svm', svm.SVC(random_state=42, kernel='rbf', probability=True))
        ]
    )
pipe.fit(X_train, y_train)
y_pipe_pred = pipe.predict(X_test)
acc = pipe.score(X_test, y_test)
print(f'ACC: {acc}')
