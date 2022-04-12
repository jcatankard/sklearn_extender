from sklearn.base import clone
import numpy


def prediction_intervals_method1():

    return

def prediction_intervals_method2(preds: numpy.ndarray, sig_level: float, samples: int, seed: int):

    return pred_range

def prediction_intervals(self, x, bootstrap_method='x', sig_level: float = 95.0,
                         samples: int = 100, seed: int = 10):
    # returns a higher and lower prediction interval
    if isinstance(x, numpy.ndarray):
        pass
    elif isinstance(x, list):
        x = numpy.array(x)
    else:
        x = x.values

    if sig_level < 50:
        raise Exception('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')
    # bootstrap method imputs
    # create preds
    if bootstrap_method == 'method1':
        pred_range = prediction_intervals_method2
    elif

    # apply range on orinignal preds

    return array2d