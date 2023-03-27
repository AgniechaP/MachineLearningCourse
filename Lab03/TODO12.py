from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import json


# 8. Wytrenuj klasyfikator bazy Iris
# Wykorzystaj wszystkie cechy
iris = load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Training a classifier
# SVC method - all features
clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
svm_acc = clf_svm.score(X_test, y_test)
print(f'svm_acc: {svm_acc}\n')

# Only 2 features
X_2 = X[:, 1]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size=0.2, stratify=y, random_state=42)
clf_svm_2 = svm.SVC(random_state=42, kernel='rbf', probability=True)
X_train_2 = X_train_2.reshape(-1, 1)
X_test_2 = X_test_2.reshape(-1, 1)
clf_svm_2.fit(X_train_2, y_train_2)
y_pred_svm_2 = clf_svm_2.predict(X_test_2)
svm_acc_2 = clf_svm_2.score(X_test_2, y_test_2)
print(f'svm_acc_2: {svm_acc_2}\n')
