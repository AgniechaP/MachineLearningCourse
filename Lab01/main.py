import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import make_classification
import pandas as pd

digits = datasets.load_digits()
print(digits)


# 2. Hello world of machine learning
# Wyswietlic jedno ze zdjec jako macierz numpy korzystajac z biblioteki matplotlib
plt.matshow(digits.images[0])
plt.show()
# Wyswietlic po kilka elementow z kazdej z dostepnych klas
fig, axs = plt.subplots(len(digits['target_names']),5) #plt.subplots(l. wierszy, l. kolumn)
for class_n in digits['target_names']:
    for col in range(5):
        axs[class_n][col].imshow(digits['images'][digits['target']==class_n][col], cmap='gray_r')
        axs[class_n][col].axis('off')
plt.show()
# 3. Train / test split
# With random_state=42 , we get the same train and test sets across different executions (run). 42 is a reference to Hitchhikers guide to galaxy book (joke)
# Ustawienie random_state na wartosc int, czyli wprowadzenie ziarna o stalej wartosci pozwala zachowac odtwarzalnosc doswiadczen
X, y = digits['data'], digits['target'] # digits['target'] - etykiety klas
# Rozdzielenie tablic X i y na 25% danych tworzących dane testowe i 75% danych uczących
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(len(X))
print(len(X_train))
print(len(X_test))

# 4. Olivetti faces
# The stratify parameter asks whether you want to retain the same proportion of classes in the train and test sets that are found in the entire original dataset.
# Laod database
faces = fetch_olivetti_faces()
print('Olivetti faces keys: ', faces.keys())
X_faces, y_faces = faces['images'], faces['target']
# Split to train (80%) and test (20%)
# X_faces_train, X_faces_test, y_faces_train, y_faces_test = train_test_split(X_faces, y_faces, test_size=0.20, random_state=42, stratify=y_faces)

# Split to 50% train and 50% test and show images from test with targets
X_faces_train, X_faces_test, y_faces_train, y_faces_test = train_test_split(X_faces, y_faces, test_size=0.50, random_state=42, stratify=y_faces)
# Show faces from test dataset - unique
i = 0
fig_f, axs_f = plt.subplots(len(np.unique(y_faces_test))//5,5)
for class_f in range(len(np.unique(y_faces_test))//5):
    for col_f in range(5):
        axs_f[class_f][col_f].matshow(X_faces_test[i])
        axs_f[class_f][col_f].set_title(y_faces_test[i])
        i += 1
        axs_f[class_f][col_f].axis('off')
plt.show()

# 5. Different database
from sklearn.datasets import load_iris
iris = load_iris()
print('Database iris keys: ', iris.keys())
print('Target names: \n')
print(list(iris['target_names']))

X_iris, y_iris = iris['data'], iris['target']
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.20, random_state=42, stratify=y_iris)
print(f'Iris train database: {len(X_train)}')

# 6. Make_classification
# Binary classification dataset - labels with only 2 possible values 0/1, so we should set n_classes to 2
# Create dataset with 1000 observations, 5 features out of which 3 will be informative and 2 redundant
X_new, y_new = make_classification(
    n_samples=1000, # 1000 observations
    n_features=5, # 5 total features
    n_informative=3, # 3 'useful' features
    n_classes=2, # binary target/label
    random_state=999
)
# Convert output of make_classification() into a pandas DataFrame (it's easier than NumPy arrays)
# Create DataFrame with features as columns
dataset_new = pd.DataFrame(X_new)
# Give names to the features
dataset_new.columns = ['X1', 'X2', 'X3', 'X4', 'X5']
# Add the label as a column - without this line we will only have X1 - X5 columns
dataset_new['y_new'] = y_new
# Plot results
dataset_new.info()

# 7.

