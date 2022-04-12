from sklearn.base import clone
import numpy


def pred_intervals_training_data(self, sig_level: float, n_trials: int, seed: int):

    model = clone(self)

    return pred_range


def pred_intervals_pred_data(self, sig_level: float, n_trials: int, seed: int):

    return pred_range


def prediction_intervals(self, bootstrap_method='training_data', sig_level: float = 95.0,
                         n_trials: int = 100, seed: int = 10):
    # returns a higher and lower prediction interval

    if sig_level < 50:
        raise Exception('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')
    if bootstrap_method not in ['training_data', 'pred_data']:
        raise Exception('bootstrap_method must be either training_data or pred_data')

    if ~isinstance(seed, int):
        raise Exception('seed must be integer')

    if ~isinstance(n_trials, int):
        raise Exception('n_trials must be integer')

    if bootstrap_method == 'training_data':
        pred_range = pred_intervals_training_data(sig_level, n_trials, seed)

    elif bootstrap_method == 'pred_data':
        pred_range = pred_intervals_pred_data(sig_level, n_trials, seed)


    return pred_range