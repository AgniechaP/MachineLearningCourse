# Cross_val_score function which trains and tests a model over multiple folds of  dataset. It gives a better understanding of model performance over the whole dataset instead of just a single train/test split
# Default - 5 folds
# It can't be used for final training. Final training should take place on all available data and tested using a set of data that has been held back from the start

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# Zaladuj baze iris
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning to Decision Tree Classifier was in ex. 7 and 8
# We create a model object with the best parameters
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)

# Test model performance using cross val score
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


