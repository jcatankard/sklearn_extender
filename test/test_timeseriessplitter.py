from sklearn_extender.timeseries_splitter import TimeSeriesSplitter
import numpy as np

array2d = np.arange(100).reshape((25, 4))

# test rolling
tss = TimeSeriesSplitter(array2d)
# tss.split(test_periods=25, min_train_periods=75, window_type='expanding')
tss.split(test_periods=5, train_periods=10, window_type='rolling')

print(tss.training_validation_indices)

tss.create_training_validation_pairs()
print(tss.training_validation_pairs)

# test expanding
tss = TimeSeriesSplitter(array2d)
# tss.split(test_periods=25, min_train_periods=75, window_type='expanding')
tss.split(test_periods=5, min_train_periods=10, window_type='expanding')

print(tss.training_validation_indices)

tss.create_training_validation_pairs()
print(tss.training_validation_pairs)

tss.plot()

# test training_validation_labels
"""
array2d = np.arange(100).reshape((25, 4))
tss = TimeSeriesSplitter(array2d)
tss.split(test_periods=5, train_periods=10, window_type='rolling')
tss.create_training_validation_labels()
print(tss.training_validation_labels)
"""