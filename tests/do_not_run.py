import numpy as np
import os
import unittest

from src.models.utils import tile_images

class TestTiler(unittest.TestCase):

    def setUp(self):
        self.size = 9
        self.size_n = 9*9
        self.size_ur = 64
        self.size_u = 72
        self.size_r = 72
        self.total = 289
        self.data = np.random.rand(1, 128, 128)

    def test_tiler(self):
        preds = np.stack(tile_images(self.data))
        preds = preds[:, 0, 1:-1, 1:-1]
        
        preds_stacked = []
        for i in range(0, self.size_n, self.size):
            preds_stacked.append(np.concatenate(preds[i:i + self.size], axis = 1))
        stacked = np.concatenate(preds_stacked, axis = 0)
            
        preds_overlap = []
        for scene in range(self.size_n, self.size_n+self.size_ur, self.size - 1):
            to_concat = np.concatenate(preds[scene:scene+ (self.size - 1)], axis = 1)
            preds_overlap.append(to_concat)    
        overlapped = np.concatenate(preds_overlap, axis = 0)
        overlapped = np.pad(overlapped, (7, 7), 'constant', constant_values = 0)
            
        preds_up = []
        for scene in range(self.size_n+self.size_ur, self.size_n+self.size_ur+self.size_r, self.size):
            to_concat = np.concatenate(preds[scene:scene+self.size], axis = 1)
            preds_up.append(to_concat)   
        up = np.concatenate(preds_up, axis = 0)
        up = np.pad(up, ((7,7), (0,0)), 'constant', constant_values = 0)
            
        preds_right = []
        for scene in range(self.size_n+self.size_ur+self.size_r, self.total, self.size - 1):
            to_concat = np.concatenate(preds[scene:scene+self.size-1], axis = 1)
            preds_right.append(to_concat)   
        right = np.concatenate(preds_right, axis = 0)
        right = np.pad(right, ((0, 0), (7, 7)), 'constant', constant_values = 0)
        
        stacked = stacked + overlapped + right + up
        stacked = stacked / 4

        self.assertEqual(stacked[7:-7, 7:-7], self.data[0, 8:-8, 8:-8])
