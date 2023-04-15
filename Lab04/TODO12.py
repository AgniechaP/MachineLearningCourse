# Feature importance - what feature was more important for predicting to which branch of decision tree some data will be putted into
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load dataset
dataset = datasets.load_iris()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Next, we’ll fit a decision tree to predict flower name
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini')

# Fit the decision tree classifier
clf = clf.fit(X_train, y_train)

# Print the feature importances
feature_importances = clf.feature_importances_

# Visualize these values using a bar chart
import seaborn as sns

# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_names = [x for _,x in sorted(zip(sorted_indices, dataset.feature_names))]
sorted_importances = feature_importances[sorted_indices]

# Create a bar plot of the feature importances
plt.figure()
sns.barplot(x=sorted_importances, y=sorted_names)
plt.show()

# Plot importance - a model must be Booster, XGBModel or dict instance so we cant use clf = DecisionTreeClassifier
# XGBClassifier is decision tree classifier
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
from xgboost import plot_importance
plot_importance(model)
plt.show()

# For comparison purposes - manual bar plot with model XGBClassifier
feature_importances_model = model.feature_importances_
sorted_indices_model = feature_importances_model.argsort()[::-1]
sorted_names_model = [x for _,x in sorted(zip(sorted_indices_model, dataset.feature_names))]
sorted_importances_model = feature_importances_model[sorted_indices_model]

# Create a bar plot of the feature importances
plt.figure()
sns.barplot(x=sorted_importances_model, y=sorted_names_model)
plt.show()
