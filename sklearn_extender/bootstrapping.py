import numpy as np
from sklearn.base import clone
import numpy


def pred_intervals_training_data(self, n_trials: int) -> np.ndarray:
    # this bootstraps the training data to create new predictions

    # original train values
    x = self.train_x
    y = self.train_y
    size = self.preds.size
    nindex = np.arange(size, dtype=np.int64)

    # clone model to prevent replacement of original fitted model
    cloned_model = clone(self)

    # create array to store results
    all_results = np.empty(self.preds * n_trials, dtype=np.float64).reshape((n_trials, self.preds))
    for i in range(n_trials):
        # randomize data
        random_index = numpy.random.choice(nindex, size=size, replace=True)
        random_x = x[random_index]
        random_y = y[random_y]
        # fit
        cloned_model.fit(random_x, random_y)
        # predict
        all_results[i] = cloned_model.predict(self.pred_x)

    return all_results


def pred_intervals_pred_data(self, n_trials: int) -> np.ndarray:
    ##### don't forget to add stratification ####
    # this bootstraps the predicted values using random sampling with replacement
    size = self.preds.size
    # create array to store results
    all_results = np.empty(size * n_trials, dtype=np.float64).reshape((n_trials, size))
    for i in range(n_trials):
        random_preds = numpy.random.choice(self.preds, size=size, replace=True)
        all_results[i] = random_preds

    return all_results


def find_prediction_intervals(results: numpy.ndarray, sig_level: float) -> np.ndarray:
    # this returns the pred intervals based on all the bootstrapped predictions
    # by taking the 1-x and x percentile based on the sum total of each individual result

    # initiate pred intervals array to be the same shape as first to row of results eg each row same length as preds
    pred_intervals = np.empty_like(results[: 2], dtype=np.float64)

    # calculate the sum of each randomized prediction
    sum_results = np.array([np.sum(i) for i in results], dtype=np.float64)

    # take the 1-x and x percentiles of this prediction
    percentile_results = np.percentile(sum_results, [1 - sig_level, sig_level], method='closest_observation')

    # find the index so the correct lower bound prediction is assigned
    loc_lower = np.where(sum_results == percentile_results[0])[0]
    pred_intervals[0] = results[loc_lower]

    # find the index so the correct upper bound prediction is assigned
    loc_upper = np.where(sum_results == percentile_results[1])[0]
    pred_intervals[1] = results[loc_upper]

    return pred_intervals


def prediction_intervals(self, bootstrap_method='training_data', sig_level: float = 95.0,
                         n_trials: int = 100) -> np.ndarray:
    # returns a higher and lower prediction interval

    if sig_level < 50:
        raise Exception('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')
    if bootstrap_method not in ['training_data', 'pred_data']:
        raise Exception('bootstrap_method must be either training_data or pred_data')

    if ~isinstance(n_trials, int):
        raise Exception('n_trials must be integer')

    if bootstrap_method == 'training_data':
        all_results = pred_intervals_training_data(n_trials)

    elif bootstrap_method == 'pred_data':
        all_results = pred_intervals_pred_data(n_trials)

    return find_prediction_intervals(all_results, sig_level)
