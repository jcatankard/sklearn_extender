from sklearn.base import clone
import numpy


def coefficients(self, labels: list, intercept: bool = True) -> dict:
    # returns dictionary with coefficients as values and labels as keys
    if not isinstance(labels, list):
        labels = list(labels)

    if len(labels) != self.train_x.shape[1]:
        raise ValueError('train has different number of columns to labels')

    coefs = dict(zip(labels, self.coef_))
    if intercept:
        coefs['intercept'] = self.intercept_

    return coefs


def generate_data_for_coef_intervals(self, labels: list, n_trials: int) -> list:
    # this bootstraps the training data to create new predictions

    # original values
    x = self.train_x
    y = self.train_y
    size = y.size
    nindex = numpy.arange(y.size, dtype=numpy.int64)

    # create list to store results
    dct_labels = labels + ['intercept']
    empty_dct = dict.fromkeys(dct_labels, float(0))
    all_results = [empty_dct.copy() for n in range(n_trials)]
    for i in range(n_trials):
        # seeding for repeatable results
        numpy.random.seed(i)
        # randomize data
        random_index = numpy.random.choice(nindex, size=size, replace=True)
        random_x = x[random_index]
        random_y = y[random_index]

        # clone model to prevent replacement of original fitted model
        cloned_model = clone(self)
        cloned_model.fit(random_x, random_y)

        # coefs
        coefs = cloned_model.coefs(labels, intercept=True)
        for k in coefs:
            all_results[i][k] = coefs[k]

    return all_results


def find_coef_conf_intervals(self, results: list, sig_level: float) -> dict:
    # this returns the confidence intervals for each coefficient

    labels = results[0].keys()
    n_trials = len(results)
    conf_intervals = dict.fromkeys(labels, [float(0), float(0)])

    for k in labels:
        coefs = numpy.arange(n_trials, dtype=numpy.float64)
        for n, r in enumerate(results):
            coefs[n] = r[k]

        intervals = list(numpy.percentile(coefs, [100 - sig_level, sig_level]))
        conf_intervals[k] = intervals

    return conf_intervals


def find_coef_pvalues(self, results: list) -> dict:
    # this returns the p values for each coefficient

    labels = results[0].keys()
    n_trials = len(results)
    conf_intervals = dict.fromkeys(labels, float(0))

    for k in labels:
        coefs = numpy.arange(n_trials, dtype=numpy.float64)
        for n in range(n_trials):
            coefs[n] = results[n][k]

        pvalue = 1 - len(coefs[coefs > 0]) / n_trials
        # inverting for negative coefs
        pvalue = pvalue if pvalue < 0.5 else 1 - pvalue
        conf_intervals[k] = pvalue

    return conf_intervals


def coef_pvalues(self, labels: list = None, sig_level: float = 95.0, n_trials: int = 10 ** 4) -> dict:
    # returns a higher and lower prediction interval

    if labels is None:
        labels = [str(i) for i in range(len(self.train_x))]
    elif not isinstance(labels, list):
        labels = list(labels)

    if len(labels) != self.train_x.shape[1]:
        raise ValueError('train has different number of columns to labels')

    if sig_level < 50:
        raise ValueError('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')

    if not isinstance(n_trials, int):
        raise TypeError('n_trials must be integer')

    all_results = generate_data_for_coef_intervals(self, labels, n_trials)

    return find_coef_pvalues(self, all_results)


def coef_confidence_intervals(self, labels: list = None, sig_level: float = 95.0, n_trials: int = 10 ** 4) -> dict:
    # returns a higher and lower prediction interval

    if labels is None:
        labels = [str(i) for i in range(len(self.train_x))]
    elif not isinstance(labels, list):
        labels = list(labels)

    if len(labels) != self.train_x.shape[1]:
        raise ValueError('train has different number of columns to labels')

    if sig_level < 50:
        raise ValueError('significance level should be between 50 and 100.'
                        'common values include 90, 95, 97.5, 99 etc.')

    if not isinstance(n_trials, int):
        raise TypeError('n_trials must be integer')

    all_results = generate_data_for_coef_intervals(self, labels, n_trials)

    return find_coef_conf_intervals(self, all_results, sig_level)
