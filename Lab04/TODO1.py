from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer
from mlxtend.plotting import plot_decision_regions
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns

# 1. Zaladuj baze danych
df = pd.read_csv("diabetes.csv")

# 1. Wyswietl przkladowe dane
# 5 pierwszych wierszy. Outcome: 1 - person has diabetes, 0 - person is not diabetic
print(df.head())
# 1. Czy w zbiorze brakuje jakichs wartosci?
print(df.isnull().sum())
# Odpowiedz: Nie ma brakujacych wartosci w datasecie, ale cechy takie jak glukoza, bloodpressure... maja wartosci zerowe, co nie powinno miec miejsca. Nalezy te wartosci zamienic medianami lub wartosciami srednimi odpowiednich kolumn

# 1. Train test na surowych danych
# Cechy
X = df.drop(['Outcome'], axis=1)
# Czy ma cukrzyce czy nie ma
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Trzy wybrane metody klasyfikacji - na surowych danych
# a) LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr_predict = lr.predict(X_test)
print('Accuracy LogisticRegression: ', accuracy_score(y_lr_predict, y_test))

# b) Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_rfc_predict = rfc.predict(X_test)
print('Accuracy RandomForestClassifier: ', accuracy_score(y_rfc_predict, y_test))

# c) Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_dt_predict = dt.predict(X_test)
print('Accuracy DecisionTreeClassifier: ', accuracy_score(y_dt_predict,y_test))

# Wyznacz wynik w przypadku odrzucenia przypadków, dla których występowały braki niektórych pomiarów?

# Uzupelnianie brakujacych danych i ponowne trenowanie klasyfikatorow
# SimpleInputer
fill_values = SimpleImputer(missing_values=0.0, strategy='mean')
X_train_SI = fill_values.fit_transform(X_train)
X_test_SI = fill_values.fit_transform(X_test)

# a) SimpleInputer LogisticRegression
y_SI_lr_predict = lr.predict(X_test_SI)
print('Accuracy LogisticRegression SimpleInputer: ', accuracy_score(y_SI_lr_predict, y_test))

# Plot results - surowe dane i po uzupelnieniu/odrzuceniu brakow - SimpleInputer i LogisticRegression
plt.hist([y_lr_predict, y_SI_lr_predict], color=['r', 'b'])
labels= ["y_lr_predict","y_SI_lr_predict"]
plt.legend(labels)
plt.ylabel('Counts')
plt.title('Surowe i uzupelnione dane - SimpleInputer, LogisticRegression')
plt.show()

# b) SimpleInputer Random Forest Classifier
y_SI_rfc_predict = rfc.predict(X_test_SI)
print('Accuracy RandomForestClassifier SimpleInputer: ', accuracy_score(y_SI_rfc_predict, y_test))

# Plot results - surowe dane i po uzupelnieniu/odrzuceniu brakow - SimpleInputer i Random Forest Classifier
plt.hist([y_rfc_predict, y_SI_rfc_predict], color=['r', 'b'])
labels= ["y_rfc_predict","y_SI_rfc_predict"]
plt.legend(labels)
plt.ylabel('Counts')
plt.title('Surowe i uzupelnione dane - SimpleInputer, Random Forest Classifier')
plt.show()

# c) SimpleInputer Decision Tree Classifier
y_SI_dt_predict = dt.predict(X_test_SI)
print('Accuracy Decision Tree Classifier SimpleInputer: ', accuracy_score(y_SI_dt_predict, y_test))

# Plot results - surowe dane i po uzupelnieniu/odrzuceniu brakow - SimpleInputer i Decision Tree Classifier
plt.hist([y_dt_predict, y_SI_dt_predict], color=['r', 'b'])
labels= ["y_dt_predict","y_SI_dt_predict"]
plt.legend(labels)
plt.ylabel('Counts')
plt.title('Surowe i uzupelnione dane - SimpleInputer, Decision Tree Classifier')
plt.show()

# IterativeImputer
imputer = IterativeImputer(random_state=42)
X_train_II = imputer.fit_transform(X_train)
X_test_II = imputer.fit_transform(X_test)

# a) IterativeImputer LogisticRegression
y_II_lr_predict = lr.predict(X_test_II)
print('Accuracy LogisticRegression IterativeImputer: ', accuracy_score(y_II_lr_predict, y_test))

# Plot

# b) IterativeImputer Random Forest Classifier
y_II_rfc_predict = rfc.predict(X_test_II)
print('Accuracy RandomForestClassifier IterativeImputer: ', accuracy_score(y_II_rfc_predict, y_test))

# Plot

# c) IterativeImputer Decision Tree Classifier
y_II_dt_predict = dt.predict(X_test_II)
print('Accuracy Decision Tree Classifier IterativeImputer: ', accuracy_score(y_II_dt_predict, y_test))

# Plot

# KNNImputer
kni_imputer = KNNImputer(n_neighbors=2, missing_values=0.0)
X_train_KNI = kni_imputer.fit_transform(X_train)
X_test_KNI = kni_imputer.fit_transform(X_test)

# a) KNNImputer LogisticRegression
y_KNI_lr_predict = lr.predict(X_test_KNI)
print('Accuracy LogisticRegression KNNImputer: ', accuracy_score(y_KNI_lr_predict, y_test))

# Plot

# b) KNNImputer Random Forest Classifier
y_KNI_rfc_predict = rfc.predict(X_test_KNI)
print('Accuracy RandomForestClassifier KNNImputer: ', accuracy_score(y_KNI_rfc_predict, y_test))

# Plot

# c) KNNImputer Decision Tree Classifier
y_KNI_dt_predict = dt.predict(X_test_KNI)
print('Accuracy Decision Tree Classifier KNNImputer: ', accuracy_score(y_KNI_dt_predict, y_test))
