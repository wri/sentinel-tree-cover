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
import itertools
import numpy

from sklearn.decomposition import PCA


def pca_fit_and_filter_pixel_list(candidate_data, reference_data, threshold):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector, for a single band.

    :param list candidate_band: A list of valid candidate data
    :param list reference_band: A list of coincident valid reference data
    :param pca_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the width of the PCA filter

    :returns: A boolean list representing the pif pixels within valid_pixels
    '''
    fitted_pca = _pca_fit_single_band(candidate_data, reference_data)
    return _pca_filter_single_band(
        fitted_pca, candidate_data, reference_data, threshold)


def _pca_fit_single_band(cand_valid, ref_valid):
    ''' Uses SK Learn PCA module to do PCA fit
    '''
    X = numpy.stack([cand_valid, ref_valid]).T

    # SK Learn PCA
    pca = PCA(n_components=2)

    # Fit the points
    pca.fit(X)

    return pca


def _numpy_array_from_2arrays(array1, array2, dtype=numpy.uint16):
    ''' Efficiently combine two 1-D arrays into a single 2-D array.

    Avoids large memory usage by creating the array using
    ``numpy.fromiter`` and then reshaping a view of the resulting
    record array.  This does the equivalent of:

        numpy.array(zip(array1, array2))

    but avoids holding a potentially large number of tuples in memory.

    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> X = _numpy_array_from_2arrays(a, b)
    >>> X
    array([[1, 4],
           [2, 5],
           [3, 6]], dtype=uint16)

    :param array array1: A 1-D numpy array.
    :param array array2: A second 1-D numpy array.
    :param data-type dtype: Data type for array elements
        (must be same for both arrays)

    :returns: A 2D numpy array combining the two input arrays
    '''
    array_dtype = [('x', dtype), ('y', dtype)]

    return numpy.fromiter(zip(array1, array2), dtype=array_dtype) \
                .view(dtype=dtype) \
                .reshape((-1, 2))


def _pca_filter_single_band(pca, cand_valid, ref_valid, threshold):
    ''' Uses SciKit Learn PCA module to transform the data and filter
    '''
    major_pca_values = _pca_transform_get_only_major_values(
        pca, cand_valid, ref_valid)

    # Filter
    pixels_pass_filter = numpy.logical_and(
        major_pca_values >= (threshold * -1), major_pca_values <= threshold)

    return pixels_pass_filter


def _pca_transform_get_only_major_values(pca, cand_valid, ref_valid):
    ''' Transforms cand_valid and ref_valid but only returns the values in the
    major eigenvector's direction (the y-values)
    '''
    X = _numpy_array_from_2arrays(cand_valid, ref_valid)
    X_trans = pca.transform(X)
    return numpy.array([x[1] for x in X_trans])
