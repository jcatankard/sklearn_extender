import sklearn_extender.coefficients
import sklearn_extender.prediction_intervals
import numpy


def model_extender(model, multiplicative_seasonality: bool = False, train_size: int = None, **kwargs):
    # creates a class that inherits from sklearn model
    # adding new functionality for:
    # multiplicative seasonality
    # bootstrapping for prediction intervals and coefficient p-values & confidence intervals
    # creating dictionary of coefficients and labels
    # fixing number of training periods

    class SKLearnExtenderClass(model):

        def __init__(self, multiplicative_seasonality: bool = False, train_size: int = None, **kwargs):

            if isinstance(multiplicative_seasonality, bool):
                self.multiplicative_seasonality = multiplicative_seasonality
            else:
                raise TypeError('multiplicative_seasonality must be a boolean value')

            if (isinstance(train_size, int) & (train_size != 0)) | isinstance(train_size, type(None)):
                self.train_size = train_size
            else:
                raise TypeError('train_size must be None or a non-zero integer')

            self.preds = None
            self.train_x = None
            self.train_y = None
            self.pred_x = None

            super().__init__(**kwargs)

        def fit(self, x, y):

            if isinstance(x, numpy.ndarray):
                pass
            elif isinstance(x, list):
                x = numpy.array(x)
            else:
                x = numpy.array(x.values)

            if isinstance(y, numpy.ndarray):
                pass
            elif isinstance(y, list):
                y = numpy.array(y)
            else:
                y = numpy.array(y.values)

            if self.train_size is not None:
                if self.train_size > 0:
                    x = x[: self.train_size]
                    y = y[: self.train_size]
                elif self.train_size < 0:
                    x = x[- self.train_size:]
                    y = y[- self.train_size:]

            self.train_x = x
            self.train_y = y

            if self.multiplicative_seasonality:
                if (x.min() < 0) | (y.min() < 0):
                    # cannot take logarithm of negative
                    raise ValueError('X and y values must be >= 0')

                # transform values for multiplicative_seasonality
                # +1 to not raise error when boolean values are 0
                x = numpy.log(x + 1)
                y = numpy.log(y + 1)

            fitted_model = super().fit(x, y)
            return fitted_model

        def predict(self, x):

            if isinstance(x, numpy.ndarray):
                pass
            elif isinstance(x, list):
                x = numpy.array(x)
            else:
                x = numpy.array(x.values)

            self.pred_x = x

            if self.multiplicative_seasonality:
                if x.min() < 0:
                    # cannot take logarithm of negative
                    raise ValueError('X values must be >= 0')

                # transform values for multiplicative_seasonality
                # +1 to not raise error when boolean values are 0
                x = numpy.log(x + 1)

            y = super().predict(x)

            # transform values back
            if self.multiplicative_seasonality:
                y = numpy.exp(y) - 1

            self.preds = y
            return y

        coefs = sklearn_extender.coefficients.coefficients
        coef_pvalues = sklearn_extender.coefficients.coef_pvalues
        coef_confidence_intervals = sklearn_extender.coefficients.coef_confidence_intervals
        prediction_intervals = sklearn_extender.prediction_intervals.prediction_intervals

    return SKLearnExtenderClass(multiplicative_seasonality, train_size, **kwargs)
