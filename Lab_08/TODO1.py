import pandas as pd
import matplotlib.pyplot as plt
from darts.models import  NaiveDrift
from darts import TimeSeries

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

# 5. Metryka MAPE




