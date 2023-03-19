import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

# Downloaded training_data.txt has a csv format
# Load file - original file is without header, so we added 'charging' and 'watching'
df = pd.read_csv('./../data/training_data.txt')
print(df)

# Scatter plot - plot with one dot for each observaion. X and y must be the same leghth
plt.scatter(df['charging'], df['watching'])
plt.xlabel('Charging')
plt.ylabel('Watching')
plt.show()

# Split dataset into test and train database
X, y = df['charging'], df['watching']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Alternatively, when file is saved as csv, load and show data:
# battery_problem_data = np.loadtxt(fname='./../data/battery_problem_data.csv', delimiter=',')
# print(f'{battery_problem_data=}')
