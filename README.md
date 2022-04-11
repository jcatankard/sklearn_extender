# sklearn_extender

- the purpose of this project is to add a little extra functionality to the sci-kit learn library that i have found useful and from which the larger community may also benefit.
- a secondary purpose is simply for me to learn how to create and publish a python package.

###  add link to github project & pypi project

# overview
### add multiplicative seasonality:
- this explores how we can use logarithmic transformations to convert linear, additive models to multiplicative ones
- this only works with linear models (e.g., not random forest)

### boostrapper:
- this explores how we can use two bootstrapping techniques to create prediction intervals for a given level of signficance
- and also to create confidence intervals & p-values for our model coefficients (where available)

### timeseries_splitter:
- it is important to validate our models with test & validation data sets and in the case of timeseries, these should respect the order of time (i.e. training sets precede validations sets)
- this is otherwise known as time series nested cross validation
