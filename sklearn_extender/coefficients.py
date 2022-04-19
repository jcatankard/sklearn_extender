from sklearn.base import clone
import numpy


def coefficients(self, labels: list, intercept: bool = True) -> dict:
    # returns dictionary with coefficients as values and labels as keys
    coefs = dict(zip(labels, self.coef_))
    if intercept:
        coefs['intercept'] = self.intercept_

    self.coefs = coefs
    return coefs


def generate_data_for_coef_intervals(self, n_trials: int) -> numpy.ndarray:
    # this bootstraps the training data to create new predictions

    # original values
    x = self.train_x
    y = self.train_y
    pred_x = self.pred_x
    size = self.preds.size
    nindex = numpy.arange(y.size, dtype=numpy.int64)

    # clone model to prevent replacement of original fitted model
    cloned_model = clone(self)

    # create array to store results
    all_results = numpy.arange(size * n_trials, dtype=numpy.float64).reshape((n_trials, size))
    for i in range(n_trials):
        # seeding for repeatable results
        numpy.random.seed(i)
        # randomize data
        random_index = numpy.random.choice(nindex, size=size, replace=True)
        random_x = x[random_index]
        random_y = y[random_index]
        # fit
        cloned_model.fit(random_x, random_y)
        # predict
        all_results[i] = cloned_model.predict(pred_x)

    return all_results


def find_coef_conf_intervals(self, sig_level: float) -> dict:
    # this returns the pred intervals based on all the bootstrapped predictions
    # by taking the 100-x and x percentile based on the sum total of each individual result

    # initiate pred intervals array to be the same shape as first to row of results eg each row same length as preds
    size = self.preds.size
    pred_intervals = numpy.arange(size * 2, dtype=numpy.float64).reshape((2, size))
    sum_preds = numpy.sum(self.preds)

    # calculate the sum of each randomized prediction
    sum_results = numpy.array([numpy.sum(i) for i in results], dtype=numpy.float64)

    # take the 1-x and x percentiles of this prediction
    percentile_results = numpy.percentile(sum_results, [100 - sig_level, sig_level])

    avg_diff_lower = numpy.abs(sum_preds - percentile_results[0]) / size
    avg_diff_upper = numpy.abs(sum_preds - percentile_results[1]) / size

    pred_intervals[0] = self.preds - avg_diff_lower
    pred_intervals[1] = self.preds + avg_diff_upper

    return pred_intervals


def find_coef_pvalues(self, sig_level: float) -> dict:
    # this returns the pred intervals based on all the bootstrapped predictions
    # by taking the 100-x and x percentile based on the sum total of each individual result

    # initiate pred intervals array to be the same shape as first to row of results eg each row same length as preds
    size = self.preds.size
    pred_intervals = numpy.arange(size * 2, dtype=numpy.float64).reshape((2, size))
    sum_preds = numpy.sum(self.preds)

    # calculate the sum of each randomized prediction
    sum_results = numpy.array([numpy.sum(i) for i in results], dtype=numpy.float64)

    # take the 1-x and x percentiles of this prediction
    percentile_results = numpy.percentile(sum_results, [100 - sig_level, sig_level])

    avg_diff_lower = numpy.abs(sum_preds - percentile_results[0]) / size
    avg_diff_upper = numpy.abs(sum_preds - percentile_results[1]) / size

    pred_intervals[0] = self.preds - avg_diff_lower
    pred_intervals[1] = self.preds + avg_diff_upper

    return pred_intervals


def coef_pvalues(self, sig_level: float = 95.0, n_trials: int = 10 ** 4) -> dict:
    # returns a higher and lower prediction interval

    if sig_level < 50:
        raise Exception('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')

    if not isinstance(n_trials, int):
        raise Exception('n_trials must be integer')

    all_results = generate_data_for_coef_intervals(self, n_trials)

    return find_coef_pvalues(self, all_results, sig_level)


def coef_confidence_intervals(self, sig_level: float = 95.0, n_trials: int = 10 ** 4) -> dict:
    # returns a higher and lower prediction interval

    if sig_level < 50:
        raise Exception('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')

    if not isinstance(n_trials, int):
        raise Exception('n_trials must be integer')

    all_results = generate_data_for_coef_intervals(self, n_trials)

    return find_coef_conf_intervals(self, all_results, sig_level)
