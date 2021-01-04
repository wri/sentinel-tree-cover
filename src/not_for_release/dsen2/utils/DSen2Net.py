from __future__ import division
from keras.models import Model, Input
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
import keras.backend as K
import numpy as np
from typing import Tuple

K.set_image_data_format('channels_first')
K.set_learning_phase(0)

def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(input_shape[-1][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model



def DSen2(d10: np.ndarray, d20: np.ndarray, model) -> np.ndarray:
    """Super resolves 20 meter bans using the DSen2 convolutional
       neural network, as specified in Lanaras et al. 2018
       https://github.com/lanha/DSen2

        Parameters:
         d10 (arr): (4, X, Y) shape array with 10 meter resolution
         d20 (arr): (6, X, Y) shape array with 20 meter resolution

        Returns:
         prediction (arr): (6, X, Y) shape array with 10 meter superresolved
                          output of DSen2 on d20 array
    """
    test = [d10, d20]
    input_shape = ((4, None, None), (6, None, None))
    prediction = _predict(test, input_shape, model, deep=False)
    return prediction


def _predict(test: np.ndarray, input_shape: Tuple, model: 'model',
             deep: bool = False, run_60: bool = False) -> np.ndarray:
    """Wrapper function around Keras.model.predict

        Parameters:
         test (arr):
         input_shape (tuple)
         model (Keras.model)
         deep (bool):
         run_60 (bool):

        Returns:
         prediction (arr): (6, X, Y) shape array with 10 meter superresolved
                          output of DSen2 on d20 array
    """
    
    prediction = model.predict(test, verbose=0, batch_size = 8)
    return prediction


def superresolve(sentinel2: np.ndarray, model) -> np.ndarray:
    """Worker function to deal with types and shapes
       to superresolve a 10-band input array

        Parameters:
         sentinel2 (arr): (:, X, Y, 10) shape array with 10 meter resolution
                          bands in indexes 0-4, and 20 meter in 4- 10

        Returns:
         superresolved (arr): (:, X, Y, 10) shape array with 10 meter 
                              superresolved output of DSen2
    """
    d10 = sentinel2[..., 0:4]
    d20 = sentinel2[..., 4:10]

    d10 = np.swapaxes(d10, 1, -1)
    d10 = np.swapaxes(d10, 2, 3)
    d20 = np.swapaxes(d20, 1, -1)
    d20 = np.swapaxes(d20, 2, 3)
    superresolved = DSen2(d10, d20, model)
    superresolved = np.swapaxes(superresolved, 1, -1)
    superresolved = np.swapaxes(superresolved, 1, 2)
    sentinel2[..., 4:10] = superresolved
    return sentinel2 # returns band IDXs 3, 4, 5, 7, 8, 9