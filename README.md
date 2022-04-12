# sklearn_extender

- the purpose of this project is to add a little extra functionality to the sci-kit learn library that I have found useful and from which the larger community may also benefit.
- a secondary purpose is simply for me to learn how to create and publish a python package.

## github
 - https://github.com/jcatankard/sklearn_extender

## pypi
 - add link to pypi project

# overview
## model extender
- expands on the functionality of regular sci-kit learn models
- adding multiplicative seasonality
- calculating prediction intervals
- coefficient confidence intervals & p-values
- returning coefficients as a dictionary
- defining the size of the training data used to fit model (useful to testing what is the optimal size)

## timeseries_splitter:
- it is important to validate our models with test & validation data sets and in the case of timeseries, these should respect the order of time (i.e. training sets precede validations sets)
- this is otherwise known as time series nested cross validation
- useful for learning how good our models are and comparing which model is best if you have many in a robust way

# timeseries_splitter
## sklearn_extender.timeseries_splitter
### class TimeSeriesSplitter(self, X, y)
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
```
from sklearn_extender.timeseries_splitter import TimeSeriesSplitter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# initialise TimeSeriesSplitter object
array2d = np.arange(4000).reshape((1000, 4))
array1d = np.arange(1000)
tss = TimeSeriesSplitter(X=array2d, y=array1d)
tss.split(test_periods=30, train_periods=365, n_validations=10)

# visualise how split looks
tss_inputs.plot()

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
### function model_extender(model, multiplicative_seasonality=False, train_size=None, **kwargs)
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
### self.coefs(labels, intercept=True)
 - returns model coefficients as a dictionary with labels as keywords
 - includes intercept if = True
 - is not compatible with models that don't have .coef_ attribute



    


