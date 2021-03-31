import numpy as np
import os
import unittest
import datetime
from src.preprocessing.indices import ndvi, evi, savi, msavi2, bi, si


class TestIndices(unittest.TestCase):

    def setUp(self):
        self.bands = ([np.full(shape = (24, 16, 16, 1), 
                          fill_value = x) for x in range(0, 10)])
        self.bands = np.concatenate(self.bands, axis = -1)

    def test_ndvi(self):
        bands_copy = np.copy(self.bands)
        bands_copy = ndvi(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = (3 - 2) / (3 + 2)
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)

    def test_evi(self):
        bands_copy = np.copy(self.bands)
        bands_copy = evi(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = 2.5 * ( (3 - 2) / (3 + (6*2) - (7.5*0) + 1))
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)

    def test_savi(self):
        bands_copy = np.copy(self.bands)
        bands_copy = savi(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = 1.5 * ( (3 - 2) / (3 + 2 + 0.5))
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)

    '''
    def test_msavi2(self):
        bands_copy = np.copy(self.bands)
        bands_copy = msavi2(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = (2 * 3 + 1 - np.sqrt( (2 * 3 + 1)**2 - 8 * (3 - 2) )) / 2
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)

    def test_bi(self):
        bands_copy = np.copy(self.bands)
        bands_copy = bi(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = (0 + 2 - 1) / (0 + 2 + 1)
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)
    '''
    def test_si(self):
        bands_copy = np.copy(self.bands)
        bands_copy = si(bands_copy)
        mean_value = np.mean(bands_copy[:, :, :, -1])
        expected_mean = np.power(( (1 - 0) * (1 - 1) * (1 - 2)), 1/3)
        is_close = np.isclose(expected_mean, mean_value)
        self.assertTrue(is_close)

