# Przeszukiwanie parametrow modelu i walidacja krzyzowa

# a) Znajdowanie parametrow Drzewa Decyzyjnego przy pomocy GridSearchCV
# code and comments from https://www.projectpro.io/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Zaladuj baze iris
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# Standard scaler - to remove outliers and scale the data by making the mean of data 0 and standard deviation as 1. We are creating object std_scl to use StandardScaler
std_slc = StandardScaler()

# PCA - Principal Component Analysis - to reduce the dimension of features by creating new features which have most of the varience of the original data
pca = decomposition.PCA()

# Decision Tree Classifier as a Machine Learning model to use GridSearchCV
dec_tree = tree.DecisionTreeClassifier()

# Using Pipeline for GridSearchCV - pipeline will helps us by passing modules one by one through GridSearchCV for which we want to get the best parameters. So we are making an object pipe to create a pipeline for all the three objects std_scl, pca and dec_tree
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

# Now we have to define the parameters that we want to optimise for these three objects.
# StandardScaler doesnot requires any parameters to be optimised by GridSearchCV.
# Principal Component Analysis requires a parameter 'n_components' to be optimised. 'n_components' signifies the number of components to keep after reducing the dimension.
# DecisionTreeClassifier requires two parameters 'criterion' and 'max_depth' to be optimised by GridSearchCV