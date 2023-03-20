import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

# 1. klasyfikator SVC na bazie TODO2 z Lab01
# Wyswietlic po kilka elementow z kazdej z dostepnych klas
digits = datasets.load_digits()
# Convert 3D data to 2D data
data = digits.images.reshape(len(digits.images),-1)
# X_train = []
# y_train = []
# for i in range(5):
#     X_train.append(data[i])
#     y_train.append(digits.target[i])
# # print(digits.target[1])
# svc = svm.SVC(gamma=0.001, C=100)
# svc.fit(X_train, y_train)
# X_test = np.array([X_train[1]])
# y_test = np.array([y_train[1]])
# y_pred = svc.predict(X_test)
# print(y_pred)
# print('Score: ', svc.score(X_test, y_test))

X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
svc = svm.SVC(gamma=0.001, C=100)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(y_pred)
print('Score: ', svc.score(X_test, y_test))

# 3. Wlasny zbior uczacy
brand_names = {'VW' : 0, 'Ford' : 1, 'Opel' :2}
print(brand_names['VW'])
age = {'20' : 20, '30' : 30, '40' : 40}
# X = [brand, age, broken]
X_car = [[brand_names['VW'], age['20'], 0],
        [brand_names['Ford'], age['20'], 0],
        [brand_names['Opel'], age['20'], 0],
        [brand_names['VW'], age['20'], 1],
        [brand_names['Ford'], age['20'], 1],
        [brand_names['Opel'], age['20'], 1],
        [brand_names['VW'], age['30'], 0],
        [brand_names['Ford'], age['30'], 0],
        [brand_names['Opel'], age['30'], 0],
        [brand_names['VW'], age['40'], 0],
        [brand_names['Ford'], age['40'], 0],
        [brand_names['Opel'], age['40'], 0]]
y_car = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]
X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(X_car, y_car, test_size=0.25, random_state=42)
svc_car = svm.SVC(gamma=0.001, C=100)
svc_car.fit(X_car_train, y_car_train)
y_pred_car = svc_car.predict(X_car_test)
print(y_pred_car)
print('Score cars: ', svc_car.score(X_car_test, y_car_test))
# Dla testowego obliczylam predykcje, dlatego jako wartosc prawdziwa w confusion matrix podaje zbior testowy y
print(f'confusion_matrix: \n{confusion_matrix(y_car_test, y_pred_car)}')

# 4. Confusion matrix
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
# y_predicted
prediction = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, prediction)
print(f'confusion_matrix digits: \n{confusion_matrix(y_test, prediction)}')
ConfusionMatrixDisplay.from_predictions(y_test, prediction)
plt.show()

for input, pred, label in zip(X_test, prediction, y_test):
  if pred != label:
          if label == 8 or label == 3:
                print(input, 'has been classified as ', pred, 'and should be ', label)

# Mean squared error
mse = metrics.mean_squared_error(y_test, prediction)
print('Mean squared error: ', mse)