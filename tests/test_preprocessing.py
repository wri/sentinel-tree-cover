import numpy as np
import os
import unittest
import datetime
from src.utils.whittaker_smoother import intialize_smoother, smooth
from src.utils.whittaker_smoother import unpacking_apply_along_axis, parallel_apply_along_axis
from timeit import default_timer as timer

class TestSmoother(unittest.TestCase):

	def setUp(self):
		self.data = np.ones((72, 10000))

	def test_time(self):
		begin = timer()
		coefmat = intialize_smoother()
		x = parallel_apply_along_axis(smooth, 0, self.data)
		end = timer()
		
		self.assertTrue((end - begin) <= 0.5)
		self.assertTrue(x.shape[0] == 72)
		self.assertTrue(x.shape[1] == 10000)