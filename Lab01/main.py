from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
# Train / test split
# With random_state=42 , we get the same train and test sets across different executions (run). 42 is a reference to Hitchhikers guide to galaxy book (joke)
# Ustawienie random_state na wartosc int, czyli wprowadzenie ziarna o stalej wartosci pozwala zachowac odtwarzalnosc doswiadczen
X, y = digits['data'], digits['target'] # digits['target'] - etykiety klas
# Rozdzielenie tablic X i y na 25% danych tworzących dane testowe i 75% danych uczących
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(len(X))
print(len(X_train))
print(len(X_test))
