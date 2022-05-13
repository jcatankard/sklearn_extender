import pandas
import numpy
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt


class TimeSeriesForecast():

    def __init__(self, multiplicative_seasonality: bool = False, train_size: int = None):

        if isinstance(multiplicative_seasonality, bool):
            self.multiplicative_seasonality = multiplicative_seasonality
        else:
            raise TypeError('multiplicative_seasonality must be a boolean value')

        if (isinstance(train_size, int) & (train_size != 0)) | isinstance(train_size, type(None)):
            self.train_size = train_size
        else:
            raise TypeError('train_size must be None or a non-zero integer')

        self.coefs = None
        self.poly_coefs = None
        self.signals = None
        self.datetime_start = None
        self.datetime_freq = None
        self.model = None
        self.check_columns = None

    def create_signals(self, n: int, max_n: int, df: pandas.DataFrame) -> numpy.array:
        # create and filter signal based on random n

        # create random possible start and end based on len n
        start_range = range(0, max_n - n + 1)
        numpy.random.seed(100)
        start = numpy.random.choice(start_range)
        end = start + n
        ynew = df['y'][start: end].values
        t = df['ds'][start: end].values

        # find and remove trend
        poly_coefs = numpy.polyfit(t, ynew, deg=1)
        trend = numpy.poly1d(poly_coefs)(t)
        ynew = ynew - trend

        # fft
        yhat = numpy.fft.rfft(ynew)
        # power spectral density
        psd = numpy.abs(yhat) / n
        # frequencies, dt is just 1 unit
        xf = numpy.fft.rfftfreq(n)

        # filter based on frequencies that are divisors of given multiplier
        multiplier = 4
        flter1 = multiplier / xf[1:] == numpy.round(multiplier / xf[1:], 0)
        flter1 = numpy.concatenate(([False], flter1))
        # create filter based on psd
        filter_value = numpy.mean(psd) + 2 * numpy.std(psd, ddof=0)
        flter2 = psd >= filter_value
        # combine filters
        flter = flter2 # flter1 * flter2

        # filter freqs, amps and phases
        frequencies = numpy.abs(xf[flter])
        amplitudes = numpy.abs(yhat) * 2 / n
        amplitudes = amplitudes[flter]
        phases = numpy.arctan2(yhat[flter].imag, yhat[flter].real) + numpy.pi / 2
        # adjust phases to t=0
        phases = phases - 2 * numpy.pi * frequencies * start

        return numpy.array([frequencies, amplitudes, phases]).T

    def fit_seasonality(self, df: pandas.DataFrame):
        # calculate seasonality trends with fast fourier transformation

        max_n = len(df)
        min_n = (max_n + 2) // 2
        numpy.random.seed(100)
        rand_ns = numpy.random.randint(low=min_n, high=max_n + 1, size=max_n, dtype=int)

        mse = numpy.inf
        for n in rand_ns:

            # create and filter signal
            signals = self.create_signals(n, max_n, df)

            # combine signals to fit
            new_signal = numpy.zeros(len(df['ds']), dtype=numpy.float64)
            for s in signals:
                # 0 = freq, 1 = amplitude, 2 = phase
                new_signal += s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * df['ds'])

            # create and add new global trend
            y_de_seasoned = df['y'] - new_signal
            poly_coefs = numpy.polyfit(df['ds'], y_de_seasoned, deg=1)
            trend = numpy.poly1d(poly_coefs)(df['ds'])
            new_signal += trend

            # evaluate fit
            mse_new = numpy.mean((new_signal - df['y']) ** 2)
            if mse_new < mse:
                mse = mse_new
                self.signals = signals
                self.poly_coefs = poly_coefs
                df['trend'] = trend

        # build signals
        print('train mse', mse)
        print(self.signals)
        for s in self.signals:
            print(1 / s[0])
            # 0 = freq, 1 = amplitude, 2 = phase
            df[str(s[0])] = s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * df['ds'])

        return df

    def fit(self, df: pandas.DataFrame):

        if not isinstance(df, pandas.DataFrame):
            raise TypeError('df must be a pandas dataframe')

        if not (('ds' in df.columns) & ('y' in df.columns)):
            raise ValueError('df must contain columns labelled ds and y')

        self.check_columns = [c for c in df.columns if c != 'y']

        df = (df
              .assign(ds=lambda x: pandas.to_datetime(x['ds']))
              .sort_values(by='ds')
              )

        if self.train_size is not None:
            df = df.tail(self.train_size)

        freq = pandas.infer_freq(df['ds'])
        self.datetime_freq = freq
        dr = pandas.date_range(df['ds'].min(), df['ds'].max(), freq=freq)
        if len(df) < len(dr):
            raise Exception('dataframe is missing datetime entries')
        if len(df) > len(dr):
            raise Exception('dataframe has too many datetime entries')

        if self.multiplicative_seasonality:
            cols = [c for c in df.columns if c != 'ds']
            if df[cols].values.min() < 0:
                # cannot take logarithm of negative
                raise ValueError('values must be >= 0')

            # transform values for multiplicative_seasonality
            # +1 to not raise error when boolean values are 0
            df[cols] = df[cols].applymap(lambda x: numpy.log(x + 1))

        # convert ds to numeric array
        self.datetime_start = df['ds'].min()
        df['ds'] = numpy.arange(len(df))

        # fft fit seasonality
        df_tofit = self.fit_seasonality(df)

        # fit model
        x = df_tofit.drop(columns=['ds', 'y'])
        model = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        fit_intercept=True, positive=False,
                        max_iter=1000, cv=5,
                        )
        model.fit(x, df_tofit['y'])
        self.model = model

        coefs = dict(zip(x.columns, model.coef_))
        coefs['intercept'] = model.intercept_
        print(coefs)
        self.coefs = coefs

    def predict(self, df: pandas.DataFrame) -> numpy.array:
        # creates predictions based on datetime array, x

        if not isinstance(df, pandas.DataFrame):
            raise TypeError('df must be a pandas dataframe')

        if 'ds' not in df.columns:
            raise ValueError('df must contain columns labelled ds and y')

        extra_cols = [c for c in df.columns if c not in self.check_columns]
        if len(extra_cols):
            raise ValueError(f'dataframe contains extra columns {extra_cols}')

        df = (df
              .assign(ds=lambda x: pandas.to_datetime(x['ds']))
              .sort_values(by='ds')
              )

        if pandas.infer_freq(df['ds']) != self.datetime_freq:
            raise ValueError('new datetime array does not match freq of fitted array')

        new_df = pandas.DataFrame()
        start = numpy.min([self.datetime_start, df['ds'].min()])
        new_df['ds'] = pandas.date_range(start, df['ds'].max(), freq=self.datetime_freq)

        # remember pred start for returning final preds
        index_pred_start = new_df['ds'][new_df['ds'] == df['ds'].min()].index[0]

        # find fit start to align signals
        index_fit_start = new_df['ds'][new_df['ds'] == self.datetime_start].index[0]
        new_df['ds'] = new_df.index - index_fit_start

        # trend
        new_df['trend'] = numpy.poly1d(self.poly_coefs)(new_df['ds'])

        # build signals
        for s in self.signals:
            # 0 = freq, 1 = amplitude, 2 = phase
            new_df[str(s[0])] = s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * new_df['ds'])

        x = (new_df
             .copy(deep=True)
             .drop(columns=['ds'])
             )
        new_df['y_preds'] = self.model.predict(x)

        # filter preds using x
        return new_df[new_df['ds'] >= index_pred_start]['y_preds']
