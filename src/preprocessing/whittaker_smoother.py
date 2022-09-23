import numpy as np
import scipy.sparse as sparse
import scipy
from scipy.sparse.linalg import splu
import multiprocessing

class Smoother:
    def __init__(self, lmbd, size, nbands = 14, dimx = 128, dimy = 128, outsize = 12, average = True):
        self.lmbd = lmbd
        self.size = size
        self.nbands = nbands
        self.dimx = dimx
        self.dimy = dimy
        self.outsize = outsize
        self.average = average
        diagonals = np.zeros(2*2+1, dtype = np.float32)
        diagonals[2] = 1.
        for i in range(2):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(2+1)
        shape = (self.size-2, self.size)
        E = sparse.eye(self.size, format = 'csc', dtype = np.float32)
        D = scipy.sparse.diags(diagonals, offsets, shape, dtype = np.float32)
        D = D.conj().T.dot(D) * self.lmbd
        coefmat = E + D
        self.splu_coef = splu(coefmat)


    def smooth(self, y: np.ndarray) -> np.ndarray:
        '''
        Apply whittaker smoothing to a 1-dimensional array, returning a 1-dimensional array
        '''
        return self.splu_coef.solve(np.array(y))


    def interpolate_array(self, x) -> np.ndarray:
        x = np.reshape(x, (self.size, self.dimx*self.dimy*self.nbands))
        x = self.smooth(x)
        x = np.reshape(x, (self.size, self.dimx, self.dimy, self.nbands))

        # median of zip(range(0, 72, 6), range(6, 72, 6))

        monthly = np.zeros((self.outsize, self.dimx, self.dimy, self.nbands), dtype = np.float32)
        index = 0
        if self.average:
            for start, end in zip(range(0, self.size + 2, self.size // self.outsize), #0, 72, 6
                                  range(self.size // self.outsize, self.size + 2, self.size // self.outsize)): # 6, 72, 6
                monthly[index] = np.mean(x[start:end], axis = 0)
                index += 1
            return monthly
        else:
            return x
