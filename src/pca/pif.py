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
from collections import namedtuple

from src.pca import pca_filter
from src.pca import robust
from src.pca import filtering

from src.pca import pixel_list_to_array
from src.pca import trim_pixel_list


pca_options = namedtuple('pca_options', 'threshold')
DEFAULT_PCA_OPTIONS = pca_options(threshold=30)

robust_options = namedtuple('robust_options', 'threshold')
DEFAULT_ROBUST_OPTIONS = robust_options(threshold=100)


def generate_mask_pifs(combined_mask):
    ''' Creates the pseudo-invariant features from the reference and candidate
    valid data masks (filtering out pixels where either the candidate or
    reference is masked, i.e. the mask value is False).

    :param array combined_mask: A 2D array representing a mask of the valid
                                pixels in both the candidate array and
                                reference array

    :returns: A 2D boolean array representing pseudo invariant features
    '''
    logging.info('PIF: Pseudo invariant feature generation is using: '
                 'Filtering using the valid data mask.')

    # Only analyse valid pixels
    valid_pixels = numpy.nonzero(combined_mask)

    pif_mask = numpy.zeros(combined_mask.shape, dtype=numpy.bool)
    pif_mask[valid_pixels] = True

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        pif_pixels = numpy.nonzero(pif_mask)
        no_pif_pixels = len(pif_pixels[0])
        no_total_pixels = combined_mask.size
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PIF: Found {} final PIFs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))

    return pif_mask


def generate_robust_pifs(candidate_band, reference_band, combined_mask,
                         parameters=DEFAULT_ROBUST_OPTIONS):
    ''' Performs a robust fit to the valid pixels and filters according
    to the distance from the fit line.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_mask: A 2D array representing a mask of the valid
                                pixels in both the candidate array and
                                reference array
    :param robust_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the distance from the fit line
                           to look for PIF pixels

    :returns: A 2D boolean array representing pseudo invariant features
    '''
    valid_pixels = numpy.nonzero(combined_mask)

    pif_pixels = generate_robust_pifs_from_pixel_list(
        candidate_band[valid_pixels], reference_band[valid_pixels], parameters)

    pif_mask = pixel_list_to_array(
        trim_pixel_list(valid_pixels, pif_pixels), candidate_band.shape)

    _info_logging(candidate_band.size, numpy.nonzero(pif_mask))

    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        _debug_logging(candidate_band, reference_band,
                       valid_pixels, numpy.nonzero(pif_mask))

    return pif_mask


def generate_robust_pifs_pixel_list(candidate_data, reference_data,
                                         parameters=DEFAULT_ROBUST_OPTIONS):
    ''' Performs a robust fit to the valid pixels and filters according
    to the distance from the fit line.

    :param list candidate_band: A list of valid candidate data
    :param list reference_band: A list of coincident valid reference data
    :param robust_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the distance from the fit line
                           to look for PIF pixels

    :returns: A boolean list representing the pif pixels within valid_pixels
    '''
    logging.info('PIF: Pseudo invariant feature generation is using: '
                 'Filtering using a robust fit.')

    # Robust fit
    gain, offset = robust.fit(candidate_data, reference_data)

    # Filter using the robust fit
    return filtering.filter_by_residuals_from_line_pixel_list(
        candidate_data, reference_data,
        threshold=parameters.threshold, line_gain=gain, line_offset=offset)


def generate_pca_pifs(candidate_band, reference_band, combined_mask,
                      parameters=DEFAULT_PCA_OPTIONS):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_mask: A 2D array representing a mask of the valid
                                pixels in both the candidate array and
                                reference array
    :param pca_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the width of the PCA filter

    :returns: A 2D boolean array representing pseudo invariant features
    '''
    valid_pixels = numpy.nonzero(combined_mask)

    pif_pixels = generate_pca_pifs_pixel_list(
        candidate_band[valid_pixels], reference_band[valid_pixels], parameters)

    pif_mask = pixel_list_to_array(
        trim_pixel_list(valid_pixels, pif_pixels), candidate_band.shape)

    _info_logging(candidate_band.size, numpy.nonzero(pif_mask))

    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        _debug_logging(candidate_band, reference_band,
                       valid_pixels, numpy.nonzero(pif_mask))

    return pif_mask


def generate_pca_pifs_pixel_list(candidate_data, reference_data,
                                 parameters=DEFAULT_PCA_OPTIONS):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param list candidate_band: A list of valid candidate data
    :param list reference_band: A list of coincident valid reference data
    :param pca_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the width of the PCA filter

    :returns: A boolean list representing the pif pixels within valid_pixels
    '''
    logging.info('PIF: Pseudo invariant feature generation is using: '
                 'Filtering using PCA.')

    return pca_filter.pca_fit_and_filter_pixel_list(
        candidate_data, reference_data, parameters)


def _info_logging(no_total_pixels, pif_pixels):
    ''' Optional logging information
    '''
    if pif_pixels[0] != [] and pif_pixels[1] != []:
        no_pif_pixels = len(pif_pixels[0])
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PIF: Found {} final PIFs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))
    else:
        logging.info('PIF: No PIF pixels found.')


def _debug_logging(c_band, r_band, valid_pixels, pif_pixels):
    ''' Optional logging information
    '''
    logging.debug('PIF: Original corrcoef = {}'.format(
        numpy.corrcoef(c_band[valid_pixels], r_band[valid_pixels])[0, 1]))

    if pif_pixels[0] != [] and pif_pixels[1] != []:
        logging.debug('PIF: Filtered corrcoef = {}'.format(
            numpy.corrcoef(c_band[pif_pixels], r_band[pif_pixels])[0, 1]))
