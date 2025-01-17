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

# 6. Find inliers
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
from sklearn import linear_model, datasets
import numpy as np
n_samples = 1000
n_outliers = 50
X, y, coef = datasets.make_regression(
    n_samples=n_samples,
    n_features=1,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=0,
)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()