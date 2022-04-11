from sklearn_extender.timeseries_splitter import TimeSeriesSplitter
import numpy as np

array2d = np.arange(4000).reshape((1000, 4))
array1d = np.arange(1000)

tss = TimeSeriesSplitter(array2d, array1d)
tss.split(test_periods=30, train_periods=365, n_validations=10, window_type='rolling')
# tss.split(test_periods=30, min_train_periods=365, n_validations=10, window_type='expanding')

print(tss.training_validation_indices)

print(tss.training_validation_data())


tss.plot()

# test training_validation_labels
"""
array2d = np.arange(100).reshape((25, 4))
array1d = np.arange(25)
tss = TimeSeriesSplitter(array2d)
tss.split(test_periods=5, train_periods=10, window_type='rolling')
tss.create_training_validation_labels()
print(tss.training_validation_labels)
"""