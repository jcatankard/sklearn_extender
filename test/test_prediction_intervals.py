from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


df = pd.DataFrame({'date': pd.date_range(start='2018-11-03', end='2022-10-01')})
df['y'] = (1 + np.sin(df['date'].dt.day) + np.sin(df['date'].dt.weekday) + np.random.rand(len(df))) * 100
df['weekday_value'] = np.sin(df['date'].dt.weekday)
df['monthday_value'] = np.sin(df['date'].dt.day)
df['random_value'] = np.random.rand(len(df))

# split into train and test
test_ln = 28
# split into train and test
train_df = (df
            .copy(deep=True)
            .head(len(df) - test_ln)
           )
train_x = (train_df
           .copy(deep=True)
           .drop(columns=['date', 'y'])
           )
train_y = train_df['y']

test_df = (df
           .copy(deep=True)
           .tail(test_ln)
          )
test_x = (test_df
          .copy(deep=True)
          .drop(columns=['date', 'y'])
          )
test_y = test_df['y']

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

sum_actuals = np.sum(test_y)
print(model.coefs(labels=list(train_x.columns)))
sum_lower = np.sum(interval_range[0])
print(round(sum_lower, 1), round(sum_lower / sum_actuals - 1, 4))
sum_preds = np.sum(preds)
print(round(sum_preds, 1), round(sum_preds / sum_actuals - 1, 4))
sum_upper = np.sum(interval_range[1])
print(round(sum_upper, 1), round(sum_upper / sum_actuals - 1, 4))

# plot results
plt.plot(test_df['date'], preds, label='preds', color='pink')
plt.fill_between(test_df['date'], (interval_range[0]), (interval_range[1]), color='blue', alpha=0.5)
# plt.plot(test_df['date'], test_y, label='actuals', color='orange')
plt.legend()
plt.ylim(bottom=0)
plt.show()

