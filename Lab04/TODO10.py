# Łączenie modeli - voting/stacking, sklearn.ensemble.VotingClassifier, sklearn.ensemble.StackingClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.ensemble import StackingClassifier


# Load Iris database
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bar plot of classes in test set
# Get data of first column in X_test (X_test is 2D array, example [[4 2 1] [2 1 3]] and it prints [4 2]
# print([i[0] for i in X_test])

# How many elements are on which class (iris type) 0, 1 or 2
counts = Counter(y_test)
print(counts)
plt.bar(counts.keys(), counts.values())
plt.show()

# Voting Classifier with soft voting
from sklearn.ensemble import VotingClassifier
def get_voting():
    # VotingClassifier based on 4 classifiers
    classificator = list()
    classificator.append(('LR', LogisticRegression(solver ='lbfgs',
                                     multi_class ='multinomial',
                                     max_iter = 200)))
    classificator.append(('SVC', SVC(gamma ='auto', probability = True)))
    classificator.append(('RFC', RandomForestClassifier()))
    classificator.append(('DTC', DecisionTreeClassifier()))
    vot_soft = VotingClassifier(estimators=classificator, voting='soft')
    return vot_soft

# Get a list of models to evaluate
def get_models():
    classificator = dict()
    classificator['LR'] = LogisticRegression(solver ='lbfgs',
                                     multi_class ='multinomial',
                                     max_iter = 200)
    classificator['SVC'] = SVC(gamma ='auto', probability = True)
    classificator['RFC'] = RandomForestClassifier()
    classificator['DTC'] = DecisionTreeClassifier()
    classificator['soft_voting'] = get_voting()
    return classificator


# Check cross_val_score of each classificator and then score of soft_voting
models = get_models()
results, names = list(), list()
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results.append(scores)
    names.append(name)
    print('>%s %.3f' % (name, np.mean(scores)))

# Box plot to show Voting Classifier has the best score among each separate classifier (LR, SVC, RFC, DTC)
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# Voting Classifier with soft voting - create final model which we can fit and predict
vot_soft = get_voting()
# Fit final model which is created by voting and soft mode
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

# Final aaccuracy_score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Soft Voting Accuracy Score",  score)

# StackingClassifier
# Create model stacking classifier
stc_class = StackingClassifier([('LR', LogisticRegression(solver ='lbfgs', multi_class ='multinomial', max_iter = 200)), \
                                ('SVC', SVC(gamma ='auto', probability = True)), \
                                ('DTC', DecisionTreeClassifier()), ('RFC', RandomForestClassifier())], final_estimator=LogisticRegression())

stc_class.fit(X_train, y_train)
y_pred_stc_class = stc_class.predict(X_test)

# Final aaccuracy_score StackingClassifier
score_stc_class = accuracy_score(y_test, y_pred_stc_class)
print("StackingClassifier Accuracy Score",  score_stc_class)
print('Score Stacing Classifier: ', stc_class.score(X_test, y_test))
