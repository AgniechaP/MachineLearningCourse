# Nonlinear regression; linear regression in Lab02

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Downloaded training_data.txt has a csv format
# Load file - original file is without header, so we added 'charging' and 'watching'
df = pd.read_csv('./../data/training_data.txt')
print(df)

# x-array - data taht we will use to make predictions - 'charging'
# y -array - data that we are trying to predict 'charging'
x = df['charging']
y = df['watching']
x = x.to_numpy()
x = x.reshape(-1, 1)

y = y.to_numpy()
y = y.reshape(-1, 1)

# Wyswietl statystyki danych
# Scatter plot - plot with one dot for each observaion. X and y must be the same leghth
plt.scatter(df['charging'], df['watching'])
plt.xlabel('Charging')
plt.ylabel('Watching')
plt.show()

# Linear regression - when variables has linear dependency between each other. We can see it plotting staticstics of data as above

# Non-linear regression - here polynominal regression
# Import PolynominalFeatures - help us transform our signal data set by adding polynominal features
from sklearn.preprocessing import PolynomialFeatures
# It is important to choose correct degree of polynominal
polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
# Converter "fits" to data, in this case, reads in every X column
# Then it "transforms" and ouputs the new polynomial data
poly_features = polynomial_converter.fit_transform(x)
poly_features.shape
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
# Fit a linear regression model on the training data
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
# Metrics of test set (data the model has never seen before)
test_predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)
ACC = model.score(X_test, y_test)
print(f'MAE: {MAE}\n MSE: {MSE}\n RMSE: {RMSE}\n ACC: {ACC}')



