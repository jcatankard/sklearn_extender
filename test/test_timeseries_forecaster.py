import matplotlib.pyplot as plt
from sklearn_extender.timeseries_forecast import TimeSeriesForecast
import pandas as pd
import numpy as np
import datetime


def create_signal(n: int) -> pd.DataFrame:
    # create time
    dt = 1
    t = np.arange(0, n, dt)

    # create signal
    weekly_seasonality = 2 * np.cos(2 * np.pi * t / 7)
    monthly_seasonality = 1 * np.sin(2 * np.pi * t / 30.5)
    quarterly_seasonality = 1 * np.sin(2 * np.pi * t / 90)
    yearly_seasonality = 5 * np.sin(2 * np.pi * t / 365.25)
    trend = 1 * np.arange(n) / n
    y_clean = weekly_seasonality + yearly_seasonality + trend + monthly_seasonality + quarterly_seasonality
    y = y_clean + 0.2 * np.random.randn(n) + 50

    df = pd.DataFrame()
    df['ds'] = pd.date_range(start=datetime.date(2020, 1, 1), periods=n)
    df['y'] = y

    return df, y_clean

# fit
train_len = 730
test_len = 730
df, y_clean = create_signal(train_len + test_len)
model = TimeSeriesForecast()
model.fit(df.head(train_len))

# predict
df['pred'] = model.predict(df.drop(columns=['y']))
rmse = np.mean((df['pred'] - df['y']) ** 2) ** 0.5
print('rmse', rmse)

plt.plot(df['ds'], df['y'], label='y')
plt.plot(df['ds'][: train_len], df['pred'][: train_len], label='train')
plt.plot(df['ds'], np.where(df.index < train_len, np.nan, df['pred']), label='pred')
plt.legend()
plt.ylim(0, df['y'].max() + 10)
plt.show()
