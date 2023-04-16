import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import missingno as msno

'''
TODO 1 Zaladuj baze danych Titanic, wykorzystaj pandas.DataFrame i metode info oraz describe
'''
original_data = pd.read_csv('titanic.csv')

def zad1():
    global original_data
    table = pd.DataFrame(data=original_data)
    # verbose = True to print all info
    info = table.info(verbose=True)
    print('Info: ', info)
    describe = table.describe()
    print('Opis: ', describe)

'''
TODO 2 Usun kolumny boat i body oraz home.dest
'''
def zad2():
    global original_data
    original_data.drop(['boat', 'body', 'home.dest', 'cabin'], axis='columns', inplace=True)
    original_data = original_data.rename(columns={'pclass': 'TicketClass'})
    print(original_data)

'''
TODO 3 Podziel dane na zbiór treningowy i testowy. Ustalmy wartość random_state na 42, użyjmy stratyfikację, test_size na poziomie 0.1
'''
def zad3():
    zad2()
    global X, y
    X = original_data.drop(['survived'], axis='columns')
    y = original_data['survived']
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1)

'''
TODO 4 Opracuj metodę, która dla danych testowych wyznacza szansę przeżycia (np. w sposób losowy).
'''
def zad4():
    zad3()
    global X, y, X_train, X_test, y_train, y_test

    def generate_random(i):
        return np.random.random() < 0.38

    results = []
    for i in X_test.iterrows():
        results.append(generate_random(i))
    print(f'Skuteczność: {sum(results == y_test) / len(results) * 100}')

'''
TODO 5 Przeszkuj bazę danych pod kątem brakujących wartości. Wyznacz ile dokładnie brakuje.
'''

def zad5():
    zad3()
    global X, y, X_train, X_test, y_train, y_test
    X = X.replace('?', np.NaN)
    msno.matrix(X)
    plt.show()
    # Result: Brakuje wartości w Age oraz Cabin

'''
TODO 6 Bazując na wiedzy z poprzednich zajęć spróbuj uzupełnić brakujące wartości.
'''
def zad6():
    zad5()
    # Fare - tylko jedna bakująca wartość - poszukaj podobnych lub usuń wiersz.
    original_data.drop(1227, inplace=True)
    # W Cabin brakuje wiekszosci danych dlatego usuwamy te kolumne - zad2()
    # Do dokonczenia - age
def main():
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    # zad5()
    zad6()
if __name__ == '__main__':
    main()



