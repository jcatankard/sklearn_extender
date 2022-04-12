from sklearn_extender.model_extender import model_extender
from sklearn.linear_model import LinearRegression
import numpy

model = model_extender(LinearRegression, multiplicative_seasonality=True, train_size=100)
print(model)
print('fit_intercept', model.fit_intercept)
print('positive', model.positive)

train_x = numpy.arange(10000).reshape(2000, 5)
train_y = numpy.arange(2000)
model.fit(train_x, train_y)

test_x = numpy.arange(100).reshape(20, 5)
preds = model.predict(test_x)
print(preds)