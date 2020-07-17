import numpy as np
import os
import unittest
import datetime

from src.downloading.utils import calculate_and_save_best_images
from src.downloading.utils import calculate_proximal_steps
from src.preprocessing.whittaker_smoother import initialize_smoother, smooth
from src.preprocessing.whittaker_smoother import unpacking_apply_along_axis, parallel_apply_along_axis
from timeit import default_timer as timer

class TestSmoother(unittest.TestCase):

	def setUp(self):
		self.data = np.ones((72, 10000))

	def test_time(self):
		begin = timer()
		coefmat = initialize_smoother()
		x = parallel_apply_along_axis(smooth, 0, self.data)
		x = np.reshape(x, (72, 10000))
		end = timer()

		print(x.shape)
		
		self.assertTrue((end - begin) <= 0.5)
		self.assertTrue(x.shape[0] == 72)
		self.assertTrue(x.shape[1] == 10000)


class TestTemporalMosaicing(unittest.TestCase):

	def setUp(self):
		self.bands = np.concatenate([np.full(shape = (1, 16, 16, 1), 
                          fill_value = x) for x in range(0, 7)], axis = 0)

		self.image_dates = np.array([0, 22, 105, 232, 295, 310, 330])

	def test_calc_and_save(self):
		bands_out, max_time = calculate_and_save_best_images(self.bands, self.image_dates)

		print(bands_out[0])
		print(bands_out.shape)
		band_means = np.mean(bands_out, axis = (1, 2, 3))

		print(band_means)

		self.assertTrue(band_means[2] == 0.5)
		self.assertTrue(band_means[6] == 1.5)
		self.assertTrue(band_means.shape[0] == 72)
		self.assertTrue(max_time == 127)


class TestProximalSteps(unittest.TestCase):

	def setUp(self):
		self.date = 3
		self.satisfactory_dates = np.array([0, 1, 2, 9, 12, 13, 14, 15])


	def test_fn(self):
		before, after = calculate_proximal_steps(self.date, 
												 self.satisfactory_dates)
		self.assertTrue(before == -1)
		self.assertTrue(after == 6)


