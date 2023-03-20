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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

digits = datasets.load_digits()
X, y = digits['data'], digits['target']
# Tworzenie obiektu klasyfikatora - tutaj drzewo decyzyjne
clf = DecisionTreeClassifier()
# fit - metoda klasyfikatora, umozliwia uczenie klasyfikatora poprzez pary danych wejsciowych X i wyjsciowych y
# fit na zbiorach train 
clf.fit(X, y)