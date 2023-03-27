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
# Loading some example data
iris = load_iris()
X, y = iris.data, iris.target
X = X[:, [0, 1]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train)
X_train_scaled = min_max_scaler.transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# Training a classifier
# SVC method
clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
clf_svm.fit(X_train_scaled, y_train)
y_pred_svm = clf_svm.predict(X_test)
svm_acc = clf_svm.score(X_test_scaled, y_test)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr = logisticRegr.fit(X_train_scaled, y_train)
y_pred_log_regr = logisticRegr.predict(X_test)
log_regr_acc = logisticRegr.score(X_test_scaled, y_test)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(X_train_scaled, y_train)
y_pred_dec_tree = decisionTree.predict(X_test)
dec_tree_acc = decisionTree.score(X_test_scaled, y_test)

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
randomTree = RandomForestClassifier()
randomTree = randomTree.fit(X_train_scaled, y_train)
y_pred_rdm_tree = randomTree.predict(X_test)
rdm_tree_acc = randomTree.score(X_test_scaled, y_test)


# Save accuracy of classifiers in dictionary
print(f'svm_acc: {svm_acc}\n Logistic Regression acc: {log_regr_acc}\n dec tree acc: {dec_tree_acc}\n Random forest acc: {rdm_tree_acc}')
data = {
    'accuracy' : [
        {
            'Method' : 'SVC',
            'Score' : svm_acc
        },
        {
            'Method' : 'LogisticRegression',
            'Score' : log_regr_acc
        },
        {
            'Method' : 'DecisionTree',
            'Score' : dec_tree_acc
        },
        {
            'Method' : 'RandomForest',
            'Score' : rdm_tree_acc
        }

    ]

}

# Save to json file
json_string = json.dumps(data)
print(json_string)
with open('json_data.json', 'w') as outfile:
    json.dump(json_string, outfile)

# Plot results - decision regions
plt.figure()
plot_decision_regions(X_test_scaled, y_test, clf=clf_svm, legend=2)
plt.figure()
plt.show()