# Random Forest Regressor - estymator typu las losowy
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Downloaded training_data.txt has a csv format
# Load file - original file is without header, so we added 'charging' and 'watching'
df = pd.read_csv('./../data/training_data.txt')
print(df)

# x-array - data taht we will use to make predictions - 'charging'
# y -array - data that we are trying to predict 'charging'
x = df['charging']
y = df['watching']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=42)
clf = RandomForestRegressor()

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
# Change X_train to array and later reshape - to solve an error expected 2D matrix
X_train = X_train.to_numpy()
X_train = X_train.reshape(-1, 1)
y_train = y_train.to_numpy()
clf.fit(X_train, y_train)

X_test = X_test.to_numpy()
X_test = X_test.reshape(-1, 1)
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
acc = clf.score(X_test, y_test)
print(f'MAE: {MAE}\n MSE: {MSE}\n RMSE: {RMSE}\n ACC: {acc}')


