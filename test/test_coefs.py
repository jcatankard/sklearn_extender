from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time


df = pd.DataFrame({'date': pd.date_range(start='2018-11-03', end='2022-10-01')})
df['y'] = (1 + np.sin(df['date'].dt.day) + np.sin(df['date'].dt.weekday) + np.random.rand(len(df))) * 100
df['weekday_value'] = np.sin(df['date'].dt.weekday)
df['monthday_value'] = np.sin(df['date'].dt.day)
df['random_value'] = np.random.rand(len(df))

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

coefs = model.coefs(labels=test_x.columns, intercept=True)
print('coefficients')
print(coefs)

t0 = time.time()
coef_pvalues = model.coef_pvalues(labels=list(test_x.columns))
t1 = time.time()
print('pvalues')
print('time', round(t1 - t0, 4))
print(coef_pvalues)

t0 = time.time()
coef_cis = model.coef_confidence_intervals(labels=list(test_x.columns))
t1 = time.time()
print('confidence intervals')
print('time', round(t1 - t0, 4))
print(coef_cis)