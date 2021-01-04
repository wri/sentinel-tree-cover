'''
Copyright 2015 Planet Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import logging
import numpy

from sklearn import linear_model


def fit(candidate_data, reference_data):
    ''' Tries a variety of robust fitting methods in what is considered
    descending order of how good the fits are with this type of data set
    (found empirically).

    :param list candidate_data: A 1D list or array representing only the image
                                data of the candidate band
    :param list reference_data: A 1D list or array representing only the image
                                data of the reference band

    :returns: A gain and an offset (tuple of floats)
    '''
    try:
        logging.debug('Robust: Trying HuberRegressor with epsilon 1.01')
        gain, offset = _huber_regressor(
            candidate_data, reference_data, 1.01)
    except:
        try:
            logging.debug('Robust: Trying HuberRegressor with epsilon 1.05')
            gain, offset = _huber_regressor(
                candidate_data, reference_data, 1.05)
        except:
            try:
                logging.debug('Robust: Trying HuberRegressor with epsilon 1.1')
                gain, offset = _huber_regressor(
                    candidate_data, reference_data, 1.1)
            except:
                try:
                    logging.debug('Robust: Trying HuberRegressor with epsilon '
                                 '1.35')
                    gain, offset = _huber_regressor(
                        candidate_data, reference_data, 1.35)
                except:
                    logging.debug('Robust: Trying RANSAC')
                    gain, offset = _ransac_regressor(
                        candidate_data, reference_data)
    return gain, offset


def _huber_regressor(candidate_data, reference_data, epsilon, max_iter=10000):
    model = linear_model.HuberRegressor(epsilon=epsilon, max_iter=max_iter)
    model.fit(numpy.array([[c] for c in candidate_data]),
              numpy.array(reference_data))
    gain = model.coef_
    offset = model.intercept_

    return gain, offset


def _ransac_regressor(candidate_data, reference_data, max_trials=10000):
    model = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                         max_trials=max_trials)
    model.fit(numpy.array([[c] for c in candidate_data]),
              numpy.array(reference_data))
    gain = model.estimator_.coef_
    offset = model.estimator_.intercept_

    return gain, offset
