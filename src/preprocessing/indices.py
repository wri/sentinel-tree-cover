import numpy as np

def grndvi(x: np.ndarray) -> np.ndarray:
    '''
    Calculates the green normalized vegetation difference index
    '''
    nir = np.clip(x[..., 3], 0, 1)
    green = np.clip(x[..., 1], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    denominator = (nir+(green+red)) + 1e-5
    return (nir-(green+red)) / denominator


def evi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    '''
    Calculates the enhanced vegetation index
    2.5 x (08 - 04) / (08 + 6 * 04 - 7.5 * 02 + 1)
    '''

    BLUE = x[..., 0]
    GREEN = x[..., 1]
    RED = x[..., 2]
    NIR = x[..., 3]
    evis = 2.5 * ( (NIR-RED) / (NIR + (6*RED) - (7.5*BLUE) + 1))
    evis = np.clip(evis, -1.5, 1.5)
    return evis


def msavi2(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    '''
    Calculates the modified soil-adjusted vegetation index 2
    (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-RED)) / 2
    '''
    BLUE = x[..., 0]
    GREEN = x[..., 1]
    RED = np.clip(x[..., 2], 0, 1)
    NIR = np.clip(x[..., 3], 0, 1)

    sqrt = (2*NIR+1)**2 - 8*(NIR-RED)
    sqrt[sqrt < 0] = 0.
    msavis = (2 * NIR + 1 - np.sqrt(sqrt)) / 2
    msavis = np.clip(msavis, -1, 1)
    return msavis


def bi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    B11 = x[..., 8]
    B4 = x[..., 2]
    B8 = x[..., 3]
    B2 = x[..., 0]
    bis = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    bis = np.clip(bis, -1, 1)
    return bis
