import numpy as np
import sys
sys.path.append('../')
from src.downloading.utils import calculate_proximal_steps
from typing import List, Any, Tuple

def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray, 
                             shadows: np.ndarray,
                             image_dates: List[int], 
                             wsize: int = 9) -> np.ndarray:
    """ Interpolates clouds and shadows for each time step with 
        linear combination of proximal clean time steps for each
        region of specified window size
        
        Parameters:
         tiles (arr):
         probs (arr): 
         shadows (arr):
         image_dates (list):
         wsize (int): 
    
        Returns:
         tiles (arr): 
    """
    
    def _fspecial_gauss(size, sigma):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g

    c_arr = np.reshape(_fspecial_gauss(wsize, ((wsize/2) - 1 ) / 2), (1, wsize, wsize, 1))
    o_arr = 1 - c_arr

    c_probs = np.copy(probs) - np.min(probs, axis = 0)
    c_probs[np.where(c_probs >= 0.33)] = 1.
    c_probs[np.where(c_probs < 0.33)] = 0.
    
    c_probs += shadows
    c_probs[np.where(c_probs >= 1.)] = 1.
    n_interp = 0
    
    
    for x in range(0, tiles.shape[1] - (wsize - 1), 1):
        for y in range(0, tiles.shape[2] - (wsize - 1), 1):
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize)/10)
            for date in range(0, tiles.shape[0]):
                if np.sum(subs[date]) >= (wsize*wsize)/10:
                    n_interp += 1
                    before, after = calculate_proximal_steps(date, satisfactory)
                    before = date + before
                    after = date + after
                    after = before if after >= tiles.shape[0] else after
                    before = after if before < 0 else before

                    before_array = tiles[before, x:x+wsize, y:y+wsize, : ]
                    after_array = tiles[after, x:x+wsize, y:y+wsize, : ]
                    original_array = tiles[np.newaxis, date, x:x+wsize, y:y + wsize, :]
                    
                    n_days_before = abs(image_dates[date] - image_dates[before])
                    n_days_after = abs(image_dates[date] - image_dates[after])
                    before_weight = 1 - n_days_before / (n_days_before + n_days_after)
                    after_weight = 1 - before_weight
                    
                    candidate = before_weight*before_array + after_weight * after_array
                    candidate = candidate * c_arr + original_array[np.newaxis] * o_arr
                    tiles[date, x:x+wsize, y:y+wsize, : ] = candidate 
                    
    print("Interpolated {} px".format(n_interp))
    return tiles