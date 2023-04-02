from sklearn.ensemble import IsolationForest
import scipy.stats
from sklearn import datasets
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split

# 5. IsolationForest
# Load dataset
dataset = datasets.fetch_openml('diabetes')#, as_frame=True, return_X_y=True)
X = dataset.data
X = X[['plas', 'mass']]
tmp = dataset.target
y = [1 if i == "tested_positive" else -1 for i in tmp]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# initializing the isolation forest
# przy contamination = 0.003, accuracy score = 0,37
# przy contamination = 0.5, accuracy score = 0,43
isolation_model = IsolationForest(contamination = 0.5)
isolation_model.fit(X_train, y_train)
y_pred = isolation_model.predict(X_test)
from sklearn.metrics import accuracy_score
# finding the accuracy
print(accuracy_score(y_pred, y_test))

# Wizualizacja wyniku predykcji
plt.hist([y_pred, y_test], color=['r', 'b'])
labels= ["y_pred", "y_test"]
plt.legend(labels)
plt.title('Porowananie y_pred i y_test')
plt.show()

