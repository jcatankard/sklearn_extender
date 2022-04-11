import numpy
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import numpy as np


def rmse_function(y_true: np.array, y_pred: np.array) -> float:
    # error function to evaluate model accuracy
    return mean_squared_error(y_true, y_pred, squared=False)


class TimeSeriesSplitter:

    def __init__(self, data):
        if isinstance(data, pandas.DataFrame):
            self.data = data.values
        elif isinstance(data, numpy.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(list)
        else:
            raise Exception(
                f'incompatible data type {type(data)}. Use numpy.ndarray, pandas.DataFrame or list objects.')

        self.rows = len(data)

    def check_split_inputs(self, test_periods: int, train_periods: int, min_train_periods: int, n_validations: int,
                           window: str):
        # raises error if these inputs are incompatible compared to the overall rows of the data

        if (train_periods is not None) & (min_train_periods is not None):
            raise Exception(
                'values cannot be assigned for both train_periods & min_train_periods'
            )

        if (train_periods is None) & (min_train_periods is None):
            raise Exception('one of train_periods & min_train_periods must be assigned')

        if window not in ['rolling', 'expanding']:
            raise Exception('training windows must either be "rolling" or "expanding"')

        if (train_periods is not None) & (window == 'expanding'):
            raise Exception(
                'if window is expanding, min_train_periods should be assigned'
                'train_periods is for a rolling window.'
            )

        if (min_train_periods is not None) & (window == 'rolling'):
            raise Exception(
                'if window is rolling, train_periods should be assigned'
                'min_train_periods is for an expanding window.'
            )

        if (n_validations is not None) & (train_periods is not None):
            if n_validations * (test_periods + train_periods) > self.rows:
                raise Exception(
                    'the data set is not large enough based on the requirements for'
                    'the number of validations and size of test_periods & train_periods.'
                )
        if (n_validations is not None) & (min_train_periods is not None):
            train_periods_values = [min_train_periods + i for i in range(n_validations)]
            if (n_validations * test_periods) + sum(train_periods_values) > self.rows:
                raise Exception(
                    'the data set is not large enough based on the requirements for'
                    'the number of validations and size of min_train_periods & train_periods.'
                )

        if min_train_periods <= 1:
            raise Exception('min_train_periods must be at least 2')
        if train_periods <= 1:
            raise Exception('train_periods must be at least 2')
        if n_validations == 0:
            raise Exception(
                'the number of validations cannot be 0.'
                'Please choose a number greater than 0 or do not select a value for this to be calculated for you'
            )

    def split(self, test_periods: int, train_periods: int = None, min_train_periods: int = None,
              n_validations: int = None, window: str = 'expanding'):
        # calculates indices for splitting timeseries data

        # validate inputs
        self.check_split_inputs(test_periods, train_periods, min_train_periods, n_validations, window)

        # calculate min train periods if not provided
        if (min_train_periods is None) & (train_periods is not None):
            min_train_periods = train_periods

        # calculate n_validations if not provided
        if n_validations is None:
            n_validations = np.int(np.floor((self.rows - min_train_periods) / test_periods))

        # configure training windows
        val_starts = list(self.datetimeseries)[-test_periods:: -test_periods][: n_validations]
        val_ends = list(self.datetimeseries)[:: -test_periods][: n_validations]
        validation_datetime_list = list(zip(val_starts, val_ends))
        validation_datetime_list.reverse()

        # adjusting min train period up so that we don't have any gaps not being used for training or validation OR fixing to train_periods input
        first_val_loc = list(self.datetimeseries).index(min(val_starts))
        min_train_periods = self.datetimeseries[0: first_val_loc].rows if train_periods is None else train_periods

        # configure training windows
        train_ends = list(self.datetimeseries)[-test_periods - 1:: -test_periods][: n_validations]

        if window == 'expanding':
            train_starts = [self.datetimeseries.min() for i in range(len(train_ends))]
        elif window == 'rolling':
            train_starts = list(self.datetimeseries)[-test_periods - min_train_periods:: -test_periods][: n_validations]

        training_datetime_list = list(zip(train_starts, train_ends))
        training_datetime_list.reverse()

        self.validation_datetimes = validation_datetime_list
        self.training_datetimes = training_datetime_list
        self.create_training_validation_pairs()
        self.create_combined_training_validation_data()
        self.final_train_periods = len(self.training_validation_pairs[-1][0])

    def create_training_validation_pairs(self) -> list:
        # return list of data tuples for each training and validation pair
        df = self.data
        training_validation_pairs = list()
        for count in range(len(self.validation_datetimes)):
            val_start, val_end = self.validation_datetimes[count]
            train_start, train_end = self.training_datetimes[count]

            training_df = df[(df.index >= train_start) & (df.index <= train_end)].copy(deep=True)
            validation_df = df[(df.index >= val_start) & (df.index <= val_end)].copy(deep=True)
            training_validation_pairs.append(tuple((training_df, validation_df)))

        self.training_validation_pairs = training_validation_pairs

    def create_combined_training_validation_data(self) -> tuple:
        # returns data with a column for each validation set flagging if the datetime is for training or validation

        df = (self.data  # leaving just the datetime index
              .drop(columns=self.data.columns)
              )
        for count in range(len(self.validation_datetimes)):
            val_start, val_end = self.validation_datetimes[count]
            train_start, train_end = self.training_datetimes[count]

            df[f'validation_{count + 1}'] = np.where((df.index >= train_start) & (df.index <= train_end), 'training',
                                                     np.where((df.index >= val_start) & (df.index <= val_end),
                                                              'validation',
                                                              None
                                                              )
                                                     )
        self.combined_training_validation_data = df

    def plot(self) -> tuple:
        # returns gantt style chart of the validation and training periods for visualisation purposes
        import matplotlib.pyplot as plt
        import matplotlib.axes as ax
        from matplotlib.patches import Patch
        plt.rcParams['figure.figsize'] = [8, 8]

        vgant_df = (pd.DataFrame(self.validation_datetimes, columns=['start', 'end'])
                    .assign(val_period=lambda x: 'validation_' + (x.index + 1).map(str),
                            task='validation',
                            )
                    )
        tgant_df = (pd.DataFrame(self.training_datetimes, columns=['start', 'end'])
                    .assign(val_period=lambda x: 'validation_' + (x.index + 1).map(str),
                            task='training',
                            )
                    )
        min_datetime = tgant_df['start'].min()
        gant_df = (vgant_df
                   .append(tgant_df)
                   .assign(periods=lambda x: x['end'] - x['start'])
                   )
        # legend
        c_dict = {'validation': '#E64646', 'training': '#E69646'}
        gant_df['color'] = gant_df['task'].apply(lambda x: c_dict[x])
        legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
        plt.legend(handles=legend_elements)

        # plot
        plt.barh(gant_df['val_period'], gant_df['periods'], left=gant_df['start'], color=gant_df['color'], linewidth=1,
                 edgecolor='black')

        # format
        plt.title('time-series nested CV plot')
        xticks = ax.Axes.get_xticks(plt.gca().axes)
        plt.xticks([xticks[0], xticks[-1]], labels=[gant_df['start'].min(), gant_df['end'].max()], visible=False)
        plt.xlim(xmin=xticks[0])
        plt.show()

    def model_validation_errors(self, model: object, y_column: str, error_function=rmse_function) -> list:
        # returns list of errors based on chosen model and error function
        # train, fit and evaluate model for each training + validation pair
        errors = list()
        base_total = 0
        cumulative_error = 0
        for training_df, validation_df in self.training_validation_pairs:
            # fit model
            train_y = training_df[y_column]
            train_x = training_df.drop(columns=[y_column])
            model.fit(train_x, train_y)

            # predict
            y_true = validation_df[y_column]
            val_x = validation_df.drop(columns=[y_column])
            y_pred = model.predict(val_x)
            errors.append(error_function(y_true, y_pred))

            base_total += np.abs(validation_df[y_column]).sum()
            cumulative_error += np.abs(np.sum(y_true) - np.sum(y_pred))

        cumulative_error_percent = 1 if base_total == 0 else cumulative_error / base_total
        return errors, cumulative_error_percent

    def find_best_model(self, models: list, y_column: str, error_function=rmse_function) -> list:
        # returns model errors, best model and coefficients from a list of models
        # find best model from list
        best_error = np.inf
        dct = {}
        for m in models:

            errors, cumulative_error_percent = self.model_validation_errors(m, y_column, error_function)
            avg_error = np.mean(errors)
            dct[str(m)] = {'avg_error': avg_error,
                           'all_errors:': errors,
                           'cumulative_error_percent': cumulative_error_percent
                           }
            if avg_error < best_error:
                dct['best_model'] = clone(m)
                best_error = avg_error
                dct['best_error'] = best_error
                dct['cumulative_error_percent'] = cumulative_error_percent

        return dct


array2d = np.arange(100).reshape((25, 4))

tss = TimeSeriesSplitter(array2d)

test = None
print(test.isnull())
