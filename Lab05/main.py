import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import missingno as msno
from sklearn import preprocessing

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
    original_data.drop(['boat', 'body', 'home.dest', 'cabin', 'name', 'sibsp', 'parch', 'embarked'], axis='columns', inplace=True)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y, test_size=0.2)

'''
TODO 4 Opracuj metodę, która dla danych testowych wyznacza szansę przeżycia (np. w sposób losowy).
'''
def zad4():
    zad3()
    global X, y, X_train, X_test, y_train, y_test

    def generate_random(i):
        return np.random.random() > 0.5

    results = []
    for i in X_test.iterrows():
        results.append(generate_random(i))
        print(generate_random(i))
    print(f'Skuteczność: {sum(results == y_test) / len(results) * 100}')


'''
TODO 5 Przeszkuj bazę danych pod kątem brakujących wartości. Wyznacz ile dokładnie brakuje.
'''

def zad5():
    zad3()
    global X, y, X_train, X_test, y_train, y_test
    global original_data
    X = X.replace('?', np.NaN)
    original_data = original_data.replace('?', np.NaN)
    X_train = X_train.replace('?', np.NaN)
    y_train = y_train.replace('?', np.NaN)
    X_test = X_test.replace('?', np.NaN)
    y_test = y_test.replace('?', np.NaN)
    msno.matrix(X)
    plt.show()
    # Result: Brakuje wartości w Age oraz Cabin

'''
TODO 6 Bazując na wiedzy z poprzednich zajęć spróbuj uzupełnić brakujące wartości - brakujące wartosci i dalsza obrobka danych tylko na zbiorze treningowym, testowego nie tykamy
'''
def zad6():
    global X, y, X_train, X_test, y_train, y_test
    zad5()
    # Fare - tylko jedna bakująca wartość - poszukaj podobnych lub usuń wiersz.
    # original_data.drop(1227, inplace=True)
    nan_fare = X_train[X_train['fare'].isna()]
    X_train.drop(nan_fare.index, inplace=True)

    # W Cabin brakuje wiekszosci danych dlatego usuwamy te kolumne - zad2()

    # Age - wyswietl histogram
    # Change Age data to float:
    X_train['age'] = pd.to_numeric(original_data['age'], downcast='float')
    plt.hist(X_train['age'])
    plt.show()

    # Uzupelnij brakujace wartosci w Age - zastepujemy NaN values srednimi
    # Find mean age value
    mean_age = X_train['age'].mean()
    # Replace NaN values in Age column with mean
    X_train['age'].fillna(value=mean_age, inplace=True)
    print('Updated col age: ', X_train['age'])

    # X_train columns: 'TicketClass', 'sex', 'age', , 'ticket', 'fare'
    '''
    TODO 7 Wykorzystaj dostępne w sklearn elementy (np. LabelEncoder) lub przygotuj własny i zamień wszystkie cechy na liczbowe
    '''
    # fit - X_train
    # transform - i na zbiorze treningowym i na testowym
    # fit znajduje wszystkie mozliwe opcje i zamienia je na wartosci liczbowe
    # transform jak dostaje zbior testowy to nie przeszukuje juz na nowo tylko przypisuje wartosci odpowiednie wytrenowanym odpowiednikom liczbowym

    # Tworzenie pustej instancji klasy LabelEncoder(), ktora nastepnie bedziemy wykorzystywac
    le = preprocessing.LabelEncoder()
    print(X_train.columns)





def main():
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    # zad5()
    zad6()
if __name__ == '__main__':
    main()



