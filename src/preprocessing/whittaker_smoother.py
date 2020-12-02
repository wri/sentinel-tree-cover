import numpy as np
import scipy.sparse as sparse
import scipy
from scipy.sparse.linalg import splu
import multiprocessing


class Smoother:

    def __init__(self, lmbd, size, nbands = 14, dim = 128):
        self.lmbd = lmbd
        self.size = size
        self.nbands = nbands
        self.dim = dim
        diagonals = np.zeros(2*2+1)
        diagonals[2] = 1.
        for i in range(2):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(2+1)
        shape = (self.size-2, self.size)
        E = sparse.eye(self.size, format = 'csc')
        D = scipy.sparse.diags(diagonals, offsets, shape)
        D = D.conj().T.dot(D) * self.lmbd
        coefmat = E + D
        self.splu_coef = splu(coefmat)

    def smooth(self, y: np.ndarray) -> np.ndarray:
        ''' 
        Apply whittaker smoothing to a 1-dimensional array, returning a 1-dimensional array
        '''
        return self.splu_coef.solve(np.array(y))


    def interpolate_array(self, x) -> np.ndarray:
        no_dem = np.delete(x, 10, -1)
        print(no_dem.shape)
        no_dem = np.reshape(no_dem, (self.size, self.dim*self.dim*self.nbands))
        no_dem = self.smooth(no_dem)
        no_dem = np.reshape(no_dem, (self.size, self.dim, self.dim, self.nbands))
        
        
        x[:, :, :, :10] = no_dem[:, :, :, :10]
        x[:, :, :, 11:] = no_dem[:, :, :, 10:]

        biweekly_dates = np.array([day for day in range(0, self.size*5, 5)])
        to_remove = np.argwhere(biweekly_dates % 15 != 0)
        x = np.delete(x, to_remove, 0)
        return x




def initialize_smoother(lmbd: int = 800, size = 72*3) -> np.ndarray:
    diagonals = np.zeros(2*2+1)
    diagonals[2] = 1.
    for i in range(2):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(2+1)
    shape = (size-2, size)
    E = sparse.eye(size, format = 'csc')
    D = scipy.sparse.diags(diagonals, offsets, shape)
    D = D.conj().T.dot(D) * lmbd
    coefmat = E + D
    splu_coef = splu(coefmat)
    return splu_coef

splu_coef = initialize_smoother(size=72*3)

def smooth(y: np.ndarray, splu_coef: np.ndarray = splu_coef) -> np.ndarray:
    ''' 
    Apply whittaker smoothing to a 1-dimensional array, returning a 1-dimensional array
    '''
    return splu_coef.solve(np.array(y))


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr)


def parallel_apply_along_axis(func1d: 'function', axis: int,
                              arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, 4)]

    pool = multiprocessing.Pool(4)
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()
    individual_results = np.concatenate(individual_results)
    if effective_axis != axis:
        individual_results = individual_results.swapaxes(axis, effective_axis)
    return np.concatenate(individual_results)


def interpolate_array(x: np.ndarray, dim: int = 128, nbands: int = 14, size = 72*3) -> np.ndarray:
    no_dem = np.delete(x, 10, -1)
    no_dem = np.reshape(no_dem, (size, dim*dim*nbands))
    no_dem = parallel_apply_along_axis(smooth, 0, no_dem)
    no_dem = np.reshape(no_dem, (size, dim, dim, nbands))
    
    
    x[:, :, :, :10] = no_dem[:, :, :, :10]
    x[:, :, :, 11:] = no_dem[:, :, :, 10:]

    biweekly_dates = np.array([day for day in range(0, size*5, 5)])
    to_remove = np.argwhere(biweekly_dates % 15 != 0)
    x = np.delete(x, to_remove, 0)
    return x