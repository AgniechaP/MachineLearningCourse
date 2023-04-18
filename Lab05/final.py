# Version 2 of laboratory instruction - tasks not in the functions - final code everything is fine

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import missingno as msno
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

'''
TODO 1 Załaduj bazę Titanic
'''
original_data = pd.read_csv('titanic.csv')
# Replace None into NaN - it's better representation and important in task 5.
original_data = original_data.replace('?', np.NaN)

# Show original_data as pandas DataFrame
table = pd.DataFrame(data=original_data)
info = table.info(verbose=True)
# print('Info: ', info)
describe = table.describe()
# print('Opis: ', describe)

'''
TODO 2 Usun wybrane kolumny
'''
original_data.drop(['boat', 'body', 'home.dest', 'cabin', 'name', 'embarked', 'ticket'], axis='columns', inplace=True)
original_data = original_data.rename(columns={'pclass': 'TicketClass'})
print(original_data.columns)

'''
TODO 3 Podziel dane na zbiór treningowy i testowy
'''
X = original_data.drop(['survived'], axis='columns')
y = original_data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y, test_size=0.2)

'''
TODO 4 Losowa szansa przeżycia 
'''
def generate_random(i):
    return np.random.random() > 0.5


results = []
for i in X_test.iterrows():
    results.append(generate_random(i))
print(f'Skuteczność: {sum(results == y_test) / len(results) * 100}')

'''
TODO 5 Przeszukaj bazę danych pod kątem brakujących wartości
'''
msno.matrix(X)
plt.show()

'''
TODO 6 Spróbuj uzupełnić brakujące wartości
'''

# Remove one row where fare value is NaN
nan_fare_y = y_train[X_train['fare'].isna()]
y_train.drop(nan_fare_y.index, inplace=True)

nan_fare = X_train[X_train['fare'].isna()]
X_train.drop(nan_fare.index, inplace=True)

# W Cabin brakuje wiekszosci danych dlatego usuwamy te kolumne - na poczatku kodu

# Show Age histogram
X_train['age'] = pd.to_numeric(X_train['age'], downcast='float')
plt.hist(X_train['age'])
plt.show()

# We want to group passengers by sex and ticket class and replace NaN values with mean
ages = X_train.groupby(['sex', 'TicketClass'])['age'].mean()
for row, passenger in X_train.loc[pd.isna(X_train['age'])].iterrows():
        X_train['age'].loc[row] = ages[passenger.sex][passenger.TicketClass]

# We should change NaN data in test dataset also to avoid errors (?)
for row, passenger in X_test.loc[pd.isna(X_test['age'])].iterrows():
        X_test['age'].loc[row] = ages[passenger.sex][passenger.TicketClass]

X_train.dropna(inplace=True)

msno.matrix(X_train)
plt.show()

'''
TODO 7 Zamieniamy cechy na liczbowe
'''
# Fit - on train dataset - it finds all options and change them to numeric values
# Transform - also on train dataset and test dataset - it doesn't find numeric analogy, only assigns values to the trained numerical counterparts

# Empty instance of LabelEncoder() class
le_ticket_class = preprocessing.LabelEncoder()
X_train['TicketClass'] = le_ticket_class.fit_transform(X_train['TicketClass'])
X_test['TicketClass'] = le_ticket_class.transform(X_test['TicketClass'])

le_sex = preprocessing.LabelEncoder()
X_train['sex'] = le_sex.fit_transform(X_train['sex'])
X_test['sex'] = le_sex.transform(X_test['sex'])

# We take together sibsp and parch into family_size
X_train['family_size'] = X_train['sibsp'] + X_train['parch']
X_train.drop(['sibsp', 'parch'], axis=1, inplace=True)
X_test['family_size'] = X_test['sibsp'] + X_test['parch']
X_test.drop(['sibsp', 'parch'], axis=1, inplace=True)


'''
TODO 8 Przygotowane dane są gotowe do wykorzystania - wytrenuj wybrany klasyfikator i oceń go względem przygotowanego wcześniej rozwiązania bazowego
'''

# Create Soft Voting Classifier
classificator = list()
classificator.append(('LR', LogisticRegression(solver='lbfgs',
                                               multi_class='multinomial',
                                               max_iter=200)))
classificator.append(('SVC', SVC(gamma='auto', probability=True)))
classificator.append(('RFC', RandomForestClassifier()))
classificator.append(('DTC', DecisionTreeClassifier()))
vot_soft = VotingClassifier(estimators=classificator, voting='soft')

# Fit model
vot_soft.fit(X_train, y_train)
y_predicted = vot_soft.predict(X_test)

# Final aaccuracy_score
from sklearn.metrics import accuracy_score
accuracy_score_score = accuracy_score(y_test, y_predicted)
print("Soft Voting Accuracy Score",  accuracy_score_score*100)

# Confusion matrix
from sklearn.metrics import confusion_matrix
Confusion_Matrix = confusion_matrix(y_test, y_predicted) # Wygenerowanie tablicy pomyłek
print('Confusion matrix: \n', Confusion_Matrix)

'''
TODO 9 Wyznacz przeżywalność w zależności od płci
'''
grouped_surv_sex = original_data.groupby(['sex'])['survived'].agg(['mean'])
print(grouped_surv_sex)

# The barplot compares the survival of men to women
plt.figure(figsize=(10,8), dpi=77)
sns.barplot(x="sex", y="survived", data=original_data)
plt.title("Survivors - Male & Female", size=17, pad=13 )
plt.show()

# How many people survived ('Survived' == 0)
survived_data=original_data['survived'].value_counts().to_frame()
print(survived_data)


'''
TODO 11 Przygotuj wykres przedstawiający korelację poszczególnych cech. Wymaga to połączenia cech z wartością oczekiwaną
'''

# Darker colors - stronger correlation between features
X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
sns.heatmap(X_combined.corr(), annot=True, cmap="coolwarm")
plt.show()

# Another idea of visualization
# sns.pairplot(X_combined.astype(float), vars=['TicketClass', 'age', 'sex', 'fare'], hue='survived')
# plt.show()
