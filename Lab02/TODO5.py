# Done from https://www.freecodecamp.org/news/how-to-build-and-train-linear-and-logistic-regression-ml-models-in-python/

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

# Downloaded training_data.txt has a csv format
# Load file - original file is without header, so we added 'charging' and 'watching'
df = pd.read_csv('./../data/training_data.txt')
print(df)

# Scatter plot - plot with one dot for each observaion. X and y must be the same leghth
# plt.scatter(df['charging'], df['watching'])
# plt.xlabel('Charging')
# plt.ylabel('Watching')
# plt.show()


# x-array - data taht we will use to make predictions - 'charging'
# y -array - data that we are trying to predict 'charging'
x = df['charging']
y = df['watching']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Import Linear Regression Estimator
from sklearn.linear_model import LinearRegression

# Create an instance of the Linear Regression Python object
model = LinearRegression()

# Change X_train to array and later reshape - to solve an error expected 2D matrix
X_train = X_train.to_numpy()
X_train = X_train.reshape(-1,1)

# Change X_test to array and later reshape
X_test = X_test.to_numpy()
X_test = X_test.reshape(-1,1)

# Train model on training data using fit
model.fit(X_train, y_train)

# Making predictions from our model, predict accepts x-array parameter and genetartes y values
predictions = model.predict(X_test)

# Plot real watching to predicted watching
plt.scatter(y_test, predictions)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

plt.hist(y_test - predictions)
plt.show()

# metrics - mean squared error
mse = metrics.mean_squared_error(y_test, predictions)
print('Mean squared error: ', mse)
# hiperparametry co to + co jestesmy w stanie zrobic zeby poprawic wyniki