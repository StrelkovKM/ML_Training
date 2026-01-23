import numpy as np
import pandas as pd
from sklearn import datasets
import scipy
dataset = datasets.load_iris()

class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        ####

        median = np.median(x, axis=0)
        return np.mean(np.abs(x - median), axis=0)

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis=0)
        self.scale = LaplaceDistribution.mean_abs_deviation_from_median(features)
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        return np.log(1/(self.scale * 2)) - (np.abs(values - self.loc) / self.scale)
        ####


    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
    

ext_target = dataset.target[:, None]
pd.DataFrame(
    np.concatenate((dataset.data, ext_target, dataset.target_names[ext_target]), axis=1),
    columns=dataset.feature_names + ['target label', 'target name'],
)

features = dataset.data
target = dataset.target

features.shape, target.shape

laplacian = LaplaceDistribution(features[:, 0])

loc0, scale0 = scipy.stats.laplace.fit(features[:, 0])
loc1, scale1 = scipy.stats.laplace.fit(features[:, 1])

# 1d case
my_distr_1 = LaplaceDistribution(features[:, 0])

# check the 1d median (loc parameter)
assert np.allclose(my_distr_1.loc, loc0), '1d distribution median error'
# check the 1d scale (loc parameter)
assert np.allclose(my_distr_1.scale, scale0), '1d distribution scale error'


# 2d case
my_distr_2 = LaplaceDistribution(features[:, :2])

# check the 2d median (loc parameter)
assert np.allclose(my_distr_2.loc, np.array([loc0, loc1])), '2d distribution median error'
# check the 2d median (loc parameter)
assert np.allclose(my_distr_2.scale, np.array([scale0, scale1])), '2d distribution scale error'

print('Seems fine!')

_test = scipy.stats.laplace(loc=[loc0, loc1], scale=[scale0, scale1])


assert np.allclose(
    my_distr_2.logpdf(features[:5, :2]),
    _test.logpdf(features[:5, :2])
), 'Logpdfs do not match scipy results!'
print('Seems fine!')
