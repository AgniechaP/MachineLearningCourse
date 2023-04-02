import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn import datasets
from sklearn import ensemble
from sklearn import impute
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline, cluster
from sklearn import decomposition, manifold

X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

# Wyswietlenie histogramu wartosci oraz boxplot dla wszystkich cech
plt.figure()
X.boxplot()
X.hist(bins=20)
plt.show()

# Wyswietlenie histogramu wartosci oraz boxplotu dla cechy mass
plt.figure()
sns.boxplot(x=X['mass'])
plt.figure()
sns.histplot(data=X, x='mass')
plt.show()
