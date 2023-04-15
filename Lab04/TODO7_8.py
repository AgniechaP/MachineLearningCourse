# Przeszukiwanie parametrow modelu i walidacja krzyzowa

# a) Znajdowanie parametrow Drzewa Decyzyjnego przy pomocy GridSearchCV
# code and comments from https://www.projectpro.io/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Zaladuj baze iris
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
# DecisionTreeClassifier -  set these two parameters as a list of values form which GridSearchCV will select the best value of parameter.

n_components = list(range(1,X.shape[1]+1,1))
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]

# Now we are creating a dictionary to set all the parameters options for different objects.
parameters = dict(pca__n_components=n_components, dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)

#Making an object clf_GS for GridSearchCV and fitting the dataset i.e X and y
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X_train, y_train)

print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])

# 8. Dodaj opcję zapisywania najlepszego modelu dla zadania związanego z grid search.
import pickle
filename = 'finalized_model.sav'

pickle.dump(clf_GS.best_estimator_, open(filename, 'wb'))
clf = pickle.load(open(filename, 'rb')) # DO wczytania
result = clf.score(X, y)
print('result of estimation best params:', result)
