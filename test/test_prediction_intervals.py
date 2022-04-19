from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import time

# import and inspect data
raw_df = pd.read_csv('daily_flights.csv')
print(raw_df.head())

# transform data
df = (raw_df
      .copy(deep=True)
      .assign(date=lambda x: pd.to_datetime(x['date']),
              month=lambda x: x['date'].dt.month,
              weekday=lambda x: x['date'].dt.weekday
              )
      .sort_values(by='date', ascending=True)
      )
weekdays = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
for id, wd in enumerate(weekdays):
    df[wd] = numpy.where(df['weekday'] == id, 1, 0)

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for id, m in enumerate(months):
    df[m] = numpy.where(df['month'] == id + 1, 1, 0)

df = df.drop(columns=['month', 'weekday'])

# split into train and test
train_df = df[df['date'].dt.year < 2022].copy(deep=True)
train_x = (train_df
           .copy(deep=True)
           .drop(columns=['date', 'flights'])
           )
train_y = train_df['flights']

test_df = df[df['date'].dt.year == 2022].copy(deep=True)
test_x = (test_df
           .copy(deep=True)
           .drop(columns=['date', 'flights'])
          )
test_y = test_df['flights']

# initiate fit and predict
model = model_extender(LinearRegression, multiplicative_seasonality=False)
model.fit(train_x, train_y)
preds = model.predict(test_x)

# create interval ranges
t0 = time.time()
interval_range = model.prediction_intervals(how='overall', sig_level=95, n_trials=10 ** 4)
# interval_range = model.prediction_intervals(how='datapoint', sig_level=95, n_trials=10 ** 4)
t1 = time.time()
print('time', round(t1 - t0, 4))

sum_actuals = numpy.sum(test_y)
print(model.coefs(labels=list(train_x.columns)))
sum_lower = numpy.sum(interval_range[0])
print(round(sum_lower, 1), round(sum_lower / sum_actuals - 1, 4))
sum_preds = numpy.sum(preds)
print(round(sum_preds, 1), round(sum_preds / sum_actuals - 1, 4))
sum_upper = numpy.sum(interval_range[1])
print(round(sum_upper, 1), round(sum_upper / sum_actuals - 1, 4))

# plot results
plt.plot(test_df['date'], preds, label='preds', color='pink')
plt.fill_between(test_df['date'], (interval_range[0]), (interval_range[1]), color='blue', alpha=0.5)
plt.plot(test_df['date'], test_y, label='actuals', color='orange')
plt.legend()
plt.show()

