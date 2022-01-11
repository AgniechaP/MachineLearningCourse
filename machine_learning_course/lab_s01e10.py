import pandas as pd
import darts
import matplotlib.pyplot as plt

from darts.models import ExponentialSmoothing, NaiveDrift, NaiveMean, AutoARIMA, RegressionModel, NaiveSeasonal, Theta
from darts.metrics import mape
from darts.utils.statistics import check_seasonality
import sklearn


def todo_1():
    print('pip install "u8darts[torch, pmdarima]"')


def todo_2():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    print(df.describe())

    df.plot()
    plt.show()

    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    print(series)
    series.plot()
    plt.show()


def todo_3():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train1, test1 = series[:-36], series[-36:]
    train2, test2 = series.split_before(pd.Timestamp('19580101'))

    train1.plot()
    test1.plot()
    plt.show()

    train2.plot()
    test2.plot()
    plt.show()


def todo_4_5():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    model = NaiveDrift()
    model.fit(train)
    prediction = model.predict(len(test))

    score = mape(actual_series=test, pred_series=prediction)
    print(f'{score=}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.show()


def todo_6():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    for m in range(2, 48):
        is_seasonal, period = check_seasonality(train, m=m, max_lag=48, alpha=.05)
        if is_seasonal:
            print(f'There is seasonality of order {period}.')

    model = NaiveSeasonal(K=12)
    model.fit(train)
    prediction = model.predict(len(test))

    score = mape(actual_series=test, pred_series=prediction)
    print(f'{score=}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.show()


def todo_6_last_question():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    model_drift = NaiveDrift()
    model_drift.fit(train)
    model_seasonal = NaiveSeasonal(K=12)
    model_seasonal.fit(train)

    prediction_drift = model_drift.predict(len(test))
    prediction_seasonal = model_seasonal.predict(len(test))

    prediction_combined = prediction_drift + prediction_seasonal - train.last_value()
    score = mape(actual_series=test, pred_series=prediction_combined)
    print(f'{score=}')

    train.plot()
    test.plot(label='test')
    prediction_combined.plot(label='prediction_combined')
    prediction_drift.plot(label='prediction_drift')
    prediction_seasonal.plot(label='prediction_seasonal')
    plt.legend()
    plt.show()


def todo_7():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    model_exponential = ExponentialSmoothing()
    model_exponential.fit(train)

    model_theta = Theta()
    model_theta.fit(train)

    prediction_exponential = model_exponential.predict(len(test))
    score = mape(actual_series=test, pred_series=prediction_exponential)
    print(f'{score=}')

    prediction_theta = model_theta.predict(len(test))
    score = mape(actual_series=test, pred_series=prediction_theta)
    print(f'{score=}')

    train.plot()
    test.plot(label='test')
    prediction_exponential.plot(label='prediction_exponential')
    prediction_theta.plot(label='prediction_theta')
    plt.legend()
    plt.show()


def main():
    todo_1()
    todo_2()
    todo_3()
    todo_4_5()
    todo_6()
    todo_7()


if __name__ == '__main__':
    main()
