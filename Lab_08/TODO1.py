import pandas as pd
import matplotlib.pyplot as plt
from darts.models import NaiveDrift, NaiveSeasonal, ExponentialSmoothing, Theta
from darts import TimeSeries
from darts.metrics import mape
from darts.utils.statistics import check_seasonality

# 2.
original_data = pd.read_csv('AirPassengers.csv')
data = pd.DataFrame(data=original_data)
print(data)
# plt.figure()
# data.plot.bar()
# plt.show()

# Convert to Timeseries
series = TimeSeries.from_dataframe(data, 'Month', '#Passengers')
series.plot()
plt.legend()
plt.show()
print(series)

# 3.
# Zbior treningowy i testowy
train, val = series.split_before(pd.Timestamp('19580101'))

# 4.
naive_drift = NaiveDrift()
naive_drift.fit(train)
prediction_naive_drift = naive_drift.predict(len(val))

# Na jednym wykresie przedstaw dane treningowe, testowe oraz prognozowane
train.plot()
val.plot()
prediction_naive_drift.plot()
plt.title('Train and val data and predict')
plt.show()

# 5. Metryka MAPE - Mean Absolute Percentage Error
mape_metric = mape(val, prediction_naive_drift)
print('MAPE Naive Drift: ', mape_metric)

# 6. Wyznaczanie sezonowości w przedziale 2-48
# max_lag default is 24, if we check range 2-48 we should set it to 48
for m in range(2, 48):
    is_seasonal, period = check_seasonality(train, m=m, max_lag=48, alpha=0.05)
    if is_seasonal:
        print("There is seasonality of order {}.".format(period))

# NaiveSesonal z ustawionym parametrem K na znalezioną wartość - K z pętli for wyżej
K = 12
naive_seasonal = NaiveSeasonal(K=12)
# Wytrenuj model
naive_seasonal.fit(train)
# Wyznacz MAPE
prediction_naive_seasonal = naive_seasonal.predict(len(val))
mape_naive_seasonal = mape(val, prediction_naive_seasonal)
print('MAPE naive seasonal: ', mape_naive_seasonal)

# 7.
# model Exponential Smoothing
exp_smooth = ExponentialSmoothing()
exp_smooth.fit(train)
prediction_exp_smooth = exp_smooth.predict(len(val))
mape_exp_smooth = mape(val, prediction_exp_smooth)
print('MAPE Experimental Smoothing: ', mape_exp_smooth)

# model Theta
theta = Theta()
theta.fit(train)
prediction_theta = theta.predict(len(val))
mape_theta = mape(val, prediction_theta)
print('MAPE Theta: ', mape_theta)








