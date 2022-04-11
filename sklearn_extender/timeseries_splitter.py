import numpy
import pandas
import matplotlib.pyplot


class TimeSeriesSplitter:
    # splits time series data into sequential train & validation pairs
    # according to required inputs for train & validation length, window type and number of validations

    def __init__(self, data):
        self.training_validation_labels = None
        self.training_validation_pairs = None
        self.training_validation_indices = None
        self.window_type = None
        self.n_validations = None
        self.min_train_periods = None
        self.training_indices = None
        self.validation_indices = None

        if isinstance(data, pandas.DataFrame):
            self.data = data.values
        elif isinstance(data, numpy.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = numpy.array(list)
        else:
            raise Exception(
                f'incompatible data type {type(data)}.'
                f' Use numpy.ndarray, pandas.DataFrame or list objects.'
            )

        self.rows = len(data)
        self.row_index = numpy.arange(self.rows)

    def validate_inputs(self, test_periods: int, train_periods: int, min_train_periods: int, n_validations: int,
                        window_type: str):
        # raises error if these inputs are incompatible compared to the overall rows of the data

        if (train_periods is not None) & (min_train_periods is not None):
            raise Exception(
                'values cannot be assigned for both train_periods & min_train_periods'
            )

        if (train_periods is None) & (min_train_periods is None):
            raise Exception('one of train_periods & min_train_periods must be assigned')

        if window_type not in ['rolling', 'expanding']:
            raise Exception('training windows must either be "rolling" or "expanding"')

        if (train_periods is not None) & (window_type == 'expanding'):
            raise Exception(
                'if window is expanding, min_train_periods should be assigned'
                ' train_periods is for a rolling window.'
            )

        if (min_train_periods is not None) & (window_type == 'rolling'):
            raise Exception(
                'if window is rolling, train_periods should be assigned'
                ' min_train_periods is for an expanding window.'
            )
        if min_train_periods is not None:
            if min_train_periods <= 1:
                raise Exception('min_train_periods must be at least 2')
        if train_periods is not None:
            if train_periods <= 1:
                raise Exception('train_periods must be at least 2')
        if n_validations is not None:
            if n_validations == 0:
                raise Exception('n_validations must be at least 1')

        if n_validations is not None:
            min_train_periods = min_train_periods if min_train_periods is not None else train_periods
            if n_validations * test_periods + min_train_periods > self.rows:
                raise Exception(
                    'the data set is not large enough based on the requirements for'
                    ' the number of validations and size of test_periods & train periods.'
                )

    def split(self, test_periods: int, train_periods: int = None, min_train_periods: int = None,
              n_validations: int = None, window_type: str = 'expanding'):
        # calculates indices for splitting timeseries data

        self.validate_inputs(test_periods, train_periods, min_train_periods, n_validations, window_type)

        # calculate min train periods if not provided
        if min_train_periods is None:
            min_train_periods = train_periods

        # calculate n_validations if not provided
        if n_validations is None:
            n_validations = numpy.int64(numpy.floor((self.rows - min_train_periods) / test_periods))

        # configure training windows
        val_starts = list(self.row_index)[-test_periods:: -test_periods][: n_validations]
        val_ends = list(self.row_index)[:: -test_periods][: n_validations]
        val_ends = [ve + 1 for ve in val_ends]
        validation_indices = list(zip(val_starts, val_ends))
        validation_indices.reverse()

        # adjusting min train period up so that we don't have any gaps not being used for training or validation
        # OR fixing to train_periods input
        first_val_loc = list(self.row_index).index(min(val_starts))
        min_train_periods = self.row_index[0: first_val_loc].size if train_periods is None else train_periods

        # configure training windows
        train_ends = list(self.row_index)[-test_periods:: -test_periods][: n_validations]

        if window_type == 'expanding':
            train_starts = [self.row_index.min() for i in range(len(train_ends))]
        elif window_type == 'rolling':
            train_starts = list(self.row_index)[-test_periods - min_train_periods:: -test_periods][: n_validations]

        training_indices = list(zip(train_starts, train_ends))
        training_indices.reverse()

        training_validation_indices = list(zip(training_indices, validation_indices))

        self.validation_indices = validation_indices
        self.training_indices = training_indices
        self.n_validations = n_validations
        self.min_train_periods = min_train_periods
        self.window_type = window_type
        self.training_validation_indices = training_validation_indices

    def create_training_validation_pairs(self):
        # return list of data tuples for each training and validation pair

        training_validation_pairs = list(range(self.n_validations))
        for count in range(self.n_validations):
            val_start, val_end = self.validation_indices[count]
            val_data = self.data[val_start: val_end]

            train_start, train_end = self.training_indices[count]
            train_data = self.data[train_start: train_end]

            training_validation_pairs[count] = tuple((train_data, val_data))

        self.training_validation_pairs = training_validation_pairs

    def create_training_validation_labels(self):
        # returns array with train & val labels for each validation set

        df = numpy.empty((self.rows, self.n_validations), dtype=str)

        for count in range(self.n_validations):
            val_start, val_end = self.validation_indices[count]
            df[val_start: val_end, count: count + 1] = 'v'

            train_start, train_end = self.training_indices[count]
            df[train_start: train_end, count: count + 1] = 't'

        self.training_validation_labels = df

    def plot(self):
        # returns gantt style chart of the validation and training periods for visualisation purposes

        # Declaring a figure "gnt"
        fig, gnt = matplotlib.pyplot.subplots()

        for i in range(self.n_validations):
            val_indices = self.validation_indices[i]
            gnt.broken_barh([(val_indices[0], val_indices[1] - val_indices[0])], (i, 1),
                            facecolors='orange'
                            )

            train_indices = self.training_indices[i]
            gnt.broken_barh([(train_indices[0], train_indices[1] - train_indices[0])], (i, 1),
                            facecolors='red'
                            )
        gnt.set_xlabel('time')
        gnt.set_ylabel('validations')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.title('timeseries nested cross validation plot')
        matplotlib.pyplot.show()
