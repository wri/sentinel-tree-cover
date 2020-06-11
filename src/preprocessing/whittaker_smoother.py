import numpy as np
import scipy.sparse as sparse
import scipy
from scipy.sparse.linalg import splu
import multiprocessing

def intialize_smoother(lmbd = 800):
    diagonals = np.zeros(2*2+1)
    diagonals[2] = 1.
    for i in range(2):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(2+1)
    shape = (70, 72)
    E = sparse.eye(72, format = 'csc')
    D = scipy.sparse.diags(diagonals, offsets, shape)
    D = D.conj().T.dot(D) * lmbd
    coefmat = E + D
    splu_coef = splu(coefmat)
    return splu_coef

splu_coef = intialize_smoother()

def smooth(y, splu_coef = splu_coef):
    ''' 
    Apply whittaker smoothing to a 1-dimensional array, returning a 1-dimensional array
    '''
    return splu_coef.solve(np.array(y))


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
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


def interpolate_array(x, dim = 128, nbands = 14):
    no_dem = np.delete(x, 10, -1)
    no_dem = np.reshape(no_dem, (72, dim*dim*nbands))
    no_dem = parallel_apply_along_axis(smooth, 0, no_dem)
    no_dem = np.reshape(no_dem, (72, dim, dim, nbands))
    
    
    x[:, :, :, :10] = no_dem[:, :, :, :10]
    x[:, :, :, 11:] = no_dem[:, :, :, 10:]

    biweekly_dates = np.array([day for day in range(0, 360, 5)])
    to_remove = np.argwhere(biweekly_dates % 15 != 0)
    x = np.delete(x, to_remove, 0)
    return x