import matplotlib.pyplot as plt
from sklearn_extender.timeseries_forecast import TimeSeriesForecast
from sklearn_extender.timeseries_splitter import TimeSeriesSplitter
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import datetime


def create_signal(n: int) -> pd.DataFrame:
    # create time
    dt = 1
    t = np.arange(0, n, dt)

    # create signal
    weekly_seasonality = 2 * np.cos(2 * np.pi * t / 7) + 7
    monthly_seasonality = 1 * np.sin(2 * np.pi * t / 30.43) + 12
    quarterly_seasonality = 1 * np.sin(2 * np.pi * t / 91.31) + 4
    yearly_seasonality = 5 * np.sin(2 * np.pi * t / 365.25) + 1
    trend = 20 * np.arange(n) / n
    y_clean = weekly_seasonality + yearly_seasonality + trend + quarterly_seasonality + monthly_seasonality
    y = y_clean + 2 * np.random.randn(n)

    df = pd.DataFrame()
    df['ds'] = pd.date_range(start=datetime.date(2020, 1, 1), periods=n)
    df['y'] = y

    return df


# create data
df = create_signal(760)

# create tss for tsf and fbp
tss = TimeSeriesSplitter(df['ds'], df['y'])
tss.split(test_periods=30, min_train_periods=400, n_validations=12, window_type='expanding')
print(f'min_train_periods: {tss.min_train_periods}')
# tss.plot()

# validate TimeSeriesForecast model
tsf_model = TimeSeriesForecast()

tsf_error = 0
count = 1
for X_train, X_val, y_train, y_val in tss.training_validation_data:

    # tsf
    train_df = pd.DataFrame()
    train_df['ds'] = X_train
    train_df['y'] = y_train
    tsf_model.fit(train_df)

    val_df = pd.DataFrame()
    val_df['ds'] = X_val
    y_pred = tsf_model.predict(val_df)
    tsf_error += mean_squared_error(y_val, y_pred, squared=False)

    print(f'{count} validation complete')
    count += 1

print(f'timeseries forecast avg error: {tsf_error / tss.n_validations}')
print(f'fbprophet avg error: {fbf_error / tss.n_validations}')

# add weekdays and months for linear and lasso models
df['month'] = df['ds'].dt.month
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']
for m in range(12):
    df[months[m]] = np.where(df['month'] == m + 1, 1, 0)

df['weekday'] = df['ds'].dt.weekday
wds = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
for wd in range(7):
    df[wds[wd]] = np.where(df['weekday'] == wd, 1, 0)

df = df.drop(columns=['month', 'weekday'])

# add linear trend
poly_coefs = np.polyfit(np.arange(len(df)), np.array(df['y']), deg=1)
df['trend'] = np.poly1d(poly_coefs)(np.arange(len(df)))

# new tss for linear lasso models
tss = TimeSeriesSplitter(df.drop(columns=['ds', 'y']), df['y'])
tss.split(test_periods=30, min_train_periods=100, n_validations=12, window_type='expanding')

# validate basic linear lasso models
linear = LinearRegression()
lasso = LassoCV()
linear_error = 0
lasso_error = 0
for X_train, X_val, y_train, y_val in tss.training_validation_data:

    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_val)
    linear_error += mean_squared_error(y_val, y_pred, squared=False)

    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_val)
    lasso_error += mean_squared_error(y_val, y_pred, squared=False)

print(f'linear forecast avg error: {linear_error / tss.n_validations}')
print(f'lasso forecast avg error: {lasso_error / tss.n_validations}')
