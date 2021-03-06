# sklearn_extender

- the purpose of this project is to add a little extra functionality to the sci-kit learn library that I have found useful and from which the larger community may also benefit.

## github
 - https://github.com/jcatankard/sklearn_extender

## pypi
 - https://pypi.org/project/sklearn-extender

# overview
## model extender:
- expands on the functionality of regular sci-kit learn models
- adding multiplicative seasonality
- calculating prediction intervals
- coefficient confidence intervals & p-values of coefficients
- returning coefficients as a dictionary
- defining the size of the training data used to fit model (useful to testing what is the optimal size)

## timeseries_splitter:
- it is important to validate our models with test & validation data sets and in the case of timeseries, these should respect the order of time (i.e. training sets precede validations sets)
- this is otherwise known as time series nested cross validation
- useful for learning how good our models are and comparing which model is best if you have many in a robust way

## timeseries_forecast:
- uses fast fourier transformation and linear regression to infer seasonality from timeseries data
- unlike other timeseries models, user to not have to specify any seasonality as the aim is to infer it entirely from the data
- model should be able to adapt to training data with partial seasons (e.g. model yearly seasonality from 9 months or 14 months of data)
- though the more training data and more seasonal cycles the better is will fit
- user can add predictors such as holidays to improve forecast
- trend, seasonality and noise can all be returned in addition to the predictions

# timeseries_splitter
## sklearn_extender.timeseries_splitter
### class: TimeSeriesSplitter(self, X, y)
#### Parameters
##### X
 - numpy.array, list, pandas.DataFrame, pandas.Series
 - can be one or two-dimensional
##### y
 - numpy.array, list, pandas.DataFrame, pandas.Series
 - must be one-dimensional

## methods
### TimeSeriesSplitter.split(self, test_periods: int, train_periods: int = None, min_train_periods: int = None, n_validations: int = None, window_type: str = 'expanding')
 - splits the timeseries splitter objects based on requirements
#### Parameters
##### test_periods
 - the length of the testing period in whichever units supplied in data.
 - the validation periods will match this length.
##### train_periods
 - only compatible with window_type='rolling'
 - the length of the training period in whichever units supplied in data
##### min_train_periods
 - only compatible with window_type='expanding'
 - the starting length of the training period in whichever units supplied in d
##### window_type
 - either 'expanding' or 'rolling'
 - in 'rolling' the training periods will be the same length for each validation set
 - for 'expanding' the training periods will increase with each validation set to make the most of the data supplied
##### n_validations
 - number of validations periods required
 - if None, this will be calculated based on other parameters
### TimeSeriesSplitter.plot(self)
 - plots a Gantt style graph to help visualise each training & validation set

## attributes
### self.rows
 - returns number of rows in timeseries data
### self.row_index
 - returns index like object to represent rows in timeseries data
### self.window_type
 - returns window type
### self.n_validations
  - returns number of validation periods
### self.min_train_periods
  - returns number of training periods in first validation set start & end
### self.training_indices
  - returns list of tuples representing the indices for each training set start & end
  - note (0, 100) is 0 to 99 inclusive
### self.validation_indices
  - returns list of tuples representing the indices for each validation set start & end
  - note (0, 100) is 0 to 99 inclusive
### self.training_validation_indices
  - returns list of tuples representing the indices for each validation set start & end
  - note (0, 100) is 0 to 99 inclusive
### TimeSeriesSplitter.training_validation_data
 - returns list of tuples containing array of values for each training and validation pair

## examples
<p float="left">
  <img src="./images/timeseries_splitter_rolling.png" width="48%" />
  <img src="/images/timeseries_splitter_expanding.png" width="48%" /> 
</p>

```
from sklearn_extender.timeseries_splitter import TimeSeriesSplitter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# initialise TimeSeriesSplitter object
array2d = np.arange(4000).reshape((1000, 4))
array1d = np.arange(1000)
tss = TimeSeriesSplitter(x=array2d, y=array1d)
tss.split(test_periods=30, min_train_periods=365, n_validations=10)

# visualise how split looks
tss.plot()

# validate model
model = LinearRegression()
total_error = 0
for X_train, X_val, y_train, y_val in tss.training_validation_data:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    total_error += mean_squared_error(y_val, y_pred, squared=False)
    
avg_error = total_error / tss.n_validations
```
# model extender
## sklearn_extender.model_extender
### function: model_extender(model, multiplicative_seasonality=False, train_size=None, **kwargs)
#### Parameters
##### model
 - accepts sci-kit learn style model that will have its functionality modified
 - not all models will benefit from the additional functionality (e.g. all are useful for linear type models but only some for random forest)
##### multiplicative_seasonality
 - if True applies np.log(x + 1) transformation to X & y values when fitting model & predicting
 - returns normalised values after predicting
 - train and test values for fitting and predicting must be >= 0
 - takes np.log(x + 1) to handle boolean (i.e. 0 or 1) values otherwise would throw error for np.log(0)
 - this is best used with linear models (e.g. not random forest)
#### train_size
 - if positive will take train_size number of rows from tail of train values (tail because it is assumed that time series are in ascending order)
 - if negative will take train_size number of rows from head of train values
#### **kwargs
 - any other key word arguments that would be passed to the sci-kit learn model (such as fit_intercept=True)
## methods
### self.coefs(labels: list, intercept=True)
 - returns model coefficients as a dictionary with labels as keywords
 - includes intercept if = True
 - is not compatible with models that don't have .coef_ attribute
### self.prediction_intervals(how='datapoint', sig_level: float = 95.0, n_trials: int = 10 ** 4)
 - returns prediction intervals for a given level of significance
 - n_trails - the higher the more accurate the prediction intervals but the longer the processing time
 - how
   - 'datapoint' is best when each individual prediction are the unit of interest (e.g., day in a time-series or price of each house)
   - 'datapoint' is also best when the predicted metric is a form of average (e.g., conversion rate, average sell price, gross margin)
   - 'overall' is best when predicted metric is 'summable' (e.g., number of items sold) and when the individual prediction values are less important than the aggregated total
     - this will result in narrower prediction intervals than 'datapoint'
### self.coef_confidence_intervals(labels: list, sig_level: float = 95.0, n_trials: int = 10 ** 4)
 - returns confidence intervals for each coefficient and intercept for a given level of significance
### self.coef_pvalues(labels: list, n_trials: int = 10 ** 4)
 - returns p-values for each coefficient and intercept

## examples
<p float="left">
  <img src="./images/pred_intervals_by_datapoint.png" width="48%" />
  <img src="/images/pred_intervals_overall.png" width="48%" /> 
</p>

```
from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# create dataframe
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

# create coefficient dictionary
coefs = model.coefs(labels=test_x.columns, intercept=True)
print('coefficients')
print(coefs)

coef_pvalues = model.coef_pvalues(labels=labels, n_trials=10 ** 4)
print('pvalues')
print(coef_pvalues)

coef_cis = model.coef_confidence_intervals(labels=labels, sig_level=95, n_trials=10 ** 4)
print('confidence intervals')
print(coef_cis)

# make predictions
preds = model.predict(test_x)

# create interval ranges
interval_range = model.prediction_intervals(how='overall', sig_level=95, n_trials=10 ** 4)

# plot results
plt.plot(test_df['date'], preds, label='preds', color='pink')
plt.fill_between(test_df['date'], (interval_range[0]), (interval_range[1]), color='blue', alpha=0.5)
# plt.plot(test_df['date'], test_y, label='actuals', color='orange')
plt.legend()
plt.ylim(bottom=0)
plt.show()
```
# timeseries_forecast
## sklearn_extender.timeseries_forecast
### class: TimeSeriesForecast(self, train_size=None, fourier_order=5)
#### Parameters
##### train_size
 - integer
 - max number of rows used to fit data
#### fourier_order
 - represents granularity of seasonality to fit
 - higher values will capture more levels of seasonality but risks over-fitting

## methods
### TimeSeriesSplitter.fit(df)
#### Parameters
##### df
 - pandas.DataFrame
 - needs to contain a datetime column labelled 'ds' and values column labelled 'y'
 - can optionally contain additional columns such as holidays to improve fit

### TimeSeriesSplitter.predict(df)
#### parameters
##### df
 - pandas.DataFrame
 - needs to contain a datetime column labelled 'ds' and of the same frequency as dataframe that was fitted
 - should contain the same columns as fitted dataframe other than 'y'

## attributes
### self.coefs
 - return coefficients as a dictionary including intercept 
### self.seasonality
 - returns numpy array of the seasonality calculated for the timeseries
### self.trend
 - returns numpy array of the overall linear trend for the timeseries
### self.noise
 - returns numpy array of what is not considered part of the seasonality or trend

## examples
<p float="left">
  <img src="./images/timeseries_forecast.png" width="100%" />
</p>

```
import matplotlib.pyplot as plt
from sklearn_extender.timeseries_forecast import TimeSeriesForecast
import pandas as pd
import numpy as np
import datetime


def create_signal(n: int) -> pd.DataFrame:
    dt = 1
    t = np.arange(0, n, dt)

    # create signal
    weekly_seasonality = 2 * np.cos(2 * np.pi * t / 7) + 7
    monthly_seasonality = 1 * np.sin(2 * np.pi * t / 30.43) + 12
    quarterly_seasonality = 1 * np.sin(2 * np.pi * t / 91.31) + 4
    yearly_seasonality = 5 * np.sin(2 * np.pi * t / 365.25) + 1
    trend = 20 * np.arange(n) / n
    y_clean = weekly_seasonality + yearly_seasonality + trend + quarterly_seasonality + monthly_seasonality
    y = y_clean + 0.5 * np.random.randn(n)

    df = pd.DataFrame()
    df['ds'] = pd.date_range(start=datetime.date(2020, 1, 1), periods=n)
    df['y'] = y

    return df


# fit
train_len = 731
test_len = 730
df = create_signal(train_len + test_len)
model = TimeSeriesForecast()
model.fit(df.head(train_len))
print(model.coefs)

# predict
df['pred'] = model.predict(df.drop(columns=['y']))
rmse = np.mean((df['pred'] - df['y']) ** 2) ** 0.5
print('rmse', rmse)

plt.plot(df['ds'], df['y'], label='y')
plt.plot(df['ds'][: train_len], df['pred'][: train_len], label='train')
plt.plot(df['ds'], np.where(df.index < train_len, np.nan, df['pred']), label='pred')

plt.plot(df['ds'][: train_len], model.seasonality, label='seasonality')
plt.plot(df['ds'][: train_len], model.trend, label='trend')
plt.plot(df['ds'][: train_len], model.noise, label='noise')

plt.legend()
plt.show()
```