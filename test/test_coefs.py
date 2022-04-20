from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import numpy
import time

model = model_extender(LinearRegression, fit_intercept=True, multiplicative_seasonality=True, train_size=100)

train_x = numpy.arange(10000).reshape(2000, 5)
train_y = numpy.arange(2000)
model.fit(train_x, train_y)

labels = ['one', 'two', 'three', 'four', 'five']
coefs = model.coefs(labels)
print('coefficients')
print(coefs)

t0 = time.time()
coef_pvalues = model.coef_pvalues(labels=labels)
t1 = time.time()
print('pvalues')
print('time', round(t1 - t0, 4))
print(coef_pvalues)

t0 = time.time()
coef_cis = model.coef_confidence_intervals(labels=labels)
t1 = time.time()
print('confidence intervals')
print('time', round(t1 - t0, 4))
print(coef_cis)
