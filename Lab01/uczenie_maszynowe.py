from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# klasyfikator w postaci drzewa decyzyjnego, i nauczymy go jak realizować prostą funkcję logiczną – bramkę AND
# X - wektory uczace, kazdy wiersz = probka uczaca i jest wektorem cech, cechy nalezy podawac zawsze w tej samej kolejnosci
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
# y - wyjscia, przyporzadkowane do konkretnych wierszy danych uczacych
y = [0, 0, 0, 1]

# Tworzenie obiektu klasyfikatora - tutaj drzewo decyzyjne
clf = DecisionTreeClassifier()

# fit - metoda klasyfikatora, umozliwia uczenie klasyfikatora poprzez pary danych wejsciowych X i wyjsciowych y
clf.fit(X, y)
# predict - po nauczeniiu klasyfikatora, uzyskujemy wektor predykcji dla danych wejsciowych
print(clf.predict([[1, 1]]))

# Klasyfikator OR
X_or = [[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]]
y_or = [0, 1, 1, 1]
clf_or = DecisionTreeClassifier()
clf_or.fit(X_or, y_or)
print(clf_or.predict([[0,0]]))

# 10. plot_tree struktura stworzonego drzewa decyzyjnego
plot_tree(clf)
plt.show()