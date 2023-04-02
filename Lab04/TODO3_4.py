import scipy.stats
from sklearn import datasets
import matplotlib.pyplot as plt

# Zaladowanie danych
X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

# Wizualizacja mass od plas
plt.scatter(X['mass'], X['plas'])
plt.axvline(x=0)
plt.axhline(y=0)
plt.title('Mass(plas)')
plt.xlabel('Mass')
plt.ylabel('Plas')
plt.show()

# 4. Z-score
X_zscore = X.apply(scipy.stats.zscore)
# Zostawia elementy, ktorych z-score jest < 3
X_filtrated = X[X_zscore < 3 ]
print(X_filtrated.describe())
plt.scatter(X_filtrated['mass'], X_filtrated['plas'])
plt.title('Mass filtrated (plas filtrated)')
plt.xlabel('Mass filtrated')
plt.ylabel('Plas filtrated')
plt.show()

