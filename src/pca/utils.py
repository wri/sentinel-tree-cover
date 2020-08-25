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
import numpy


def pixel_list_to_array(pixel_locations, shape):
    ''' Transforms a list of pixel locations into a 2D array.

    :param tuple pixel_locations: A tuple of two lists representing the x and y
        coordinates of the locations of a set of pixels (i.e. the output of
        numpy.nonzero(valid_pixels) where valid_pixels is a 2D boolean array
        representing the pixel locations)
    :param list active_pixels: A list the same length as the x and y coordinate
        lists within pixel_locations representing whether a pixel location
        should be represented in the mask or not
    :param tuple shape: The shape of the output array consisting of a tuple
        of (height, width)

    :returns: A 2-D boolean array representing active pixels
    '''
    mask = numpy.zeros(shape, dtype=numpy.bool)
    mask[pixel_locations] = True

    return mask


def trim_pixel_list(pixel_locations, active_pixels):
    ''' Trims the list of pixel locations to only the active pixels.

    :param tuple pixel_locations: A tuple of two lists representing the x and y
        coordinates of the locations of a set of pixels (i.e. the output of
        numpy.nonzero(valid_pixels) where valid_pixels is a 2D boolean array
        representing the pixel locations)
    :param list active_pixels: A list the same length as the x and y coordinate
        lists within pixel_locations representing whether a pixel location
        should be represented in the mask or not

    :returns: A tuple of two lists representing the x and y coordinates of the
        locations of active pixels
    '''
    active_pixels = numpy.nonzero(active_pixels)[0]
    return (pixel_locations[0][active_pixels],
            pixel_locations[1][active_pixels])


def combine_valid_pixel_arrays(list_of_pixel_arrays):
    ''' Combines a list of 2D boolean pixel arrays that represent locations of
    valid pixels with only the pixels that are common in all bands.

    :param list list_of_pixel_arrays: A list of 2D boolean arrays representing
        the valid pixel locations

    :returns: A 2D boolean array representing common valid pixels
    '''
    return numpy.logical_and.reduce(list_of_pixel_arrays)


def combine_valid_pixel_lists(list_of_pixel_locations):
    ''' Combines a list of valid pixel x and y locations (for a 2D array) with
    only the pixels that are in common in all bands.

    :param list list_of_pixel_locations: A list of tuples that contain two
        lists representing the x and y coordinates of the locations of valid
        pixels (i.e. the output of numpy.nonzero(valid_pixels) where
        valid_pixels is a 2D boolean array representing the valid pixel
        locations)

    :returns: A 2D boolean array representing common valid pixels
    '''

    max_x = max([max(l[0]) for l in list_of_pixel_locations])
    max_y = max([max(l[1]) for l in list_of_pixel_locations])
    shape = (max_x + 1, max_y + 1)

    list_of_pixel_arrays = [pixel_list_to_array(p, shape) for p in
                            list_of_pixel_locations]

    return numpy.nonzero(combine_valid_pixel_arrays(list_of_pixel_arrays))
