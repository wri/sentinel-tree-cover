import numpy as np

def ndvi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # (B8 - B4)/(B8 + B4)
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    ndvis = (NIR-RED) / (NIR+RED)
    if verbose:
        mins = np.min(ndvis)
        maxs = np.max(ndvis)
        if mins < -1 or maxs > 1:
            print("ndvis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, ndvis[:, :, :, np.newaxis]], axis = -1)
    return x

def evi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # 2.5 x (08 - 04) / (08 + 6 * 04 - 7.5 * 02 + 1)
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    BLUE = x[:, :, :, 0]
    evis = 2.5 * ( (NIR-RED) / (NIR + (6*RED) - (7.5*BLUE) + 1))
    evis = np.clip(evis, -1.5, 1.5)
    x = np.concatenate([x, evis[:, :, :, np.newaxis]], axis = -1)
    return x
    
def savi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # (1.5) * ((08 - 04)/ (08 + 04 + 0.5))
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    savis = 1.5 * ( (NIR-RED) / (NIR+RED +0.5))
    if verbose:
        mins = np.min(savis)
        maxs = np.max(savis)
        if mins < -1.0 or maxs > 1.0:
        	print("SAVI: {} {}".format(mins, maxs))
    x = np.concatenate([x, savis[:, :, :, np.newaxis]], axis = -1)
    return x

def msavi2(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-RED)) / 2
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    RED = np.clip(RED, 0, 1)
    NIR = np.clip(NIR, 0, 1)
    #for i in range(x.shape[0]): # if NIR is smalelr than red, then it works
    #    NIR_i = x[i, :, :, 3]   # if nir is greater than red
    #    RED_i = x[i, :, :, 2] # 2 * 0.99 + 1**2 = 8.88, -8*.99 - 0.01 = 7.92
    #    under_sqrt = (2*NIR_i+1)**2 - 8*(NIR_i-RED_i) # NIR = 0.1, Red = 0.02 = 
    #    under_sqrt = np.min(under_sqrt)
    #    if under_sqrt <= 0:
    #        location = np.argmin(under_sqrt.flatten())
    #        print(NIR_i.flatten()[location])
    #        print(RED_i.flatten()[location])
    #        print(under_sqrt.flatten()[location])
    #        print("MSAVI2 negative sqrt at: {}, {}".format(i, under_sqrt))
    msavis = (2 * NIR + 1 - np.sqrt( (2*NIR+1)**2 - 8*(NIR-RED) )) / 2
    if verbose:
        mins = np.min(msavis)
        maxs = np.max(msavis)
        if mins < -1 or maxs > 1:
            print("MSAVIS error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, msavis[:, :, :, np.newaxis]], axis = -1)
    return x

def bi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # (2 + 0 - 1) / (2 + 0 + 1)
    # This is still in the trained model, but the index is gibberish
    # as it is based on Landsat bands, not sentinel bands
    # https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    # correct bare soil index is:
    # (B11 + B4) - (B8 + B2) / (B11 + B4) + (B8 + B2)
    # Landsat: (NIR + green - red) / (NIR + green + red)
    # current: (BLUE + RED - GREEN) / (BLUE + RED + GREEN)
    BLUE = np.clip(x[:, :, :, 0], 0, 1)
    RED = np.clip(x[:, :, :, 2], 0, 1)
    GREEN = np.clip(x[:, :, :, 1], 0, 1)
    bis = (BLUE + RED - GREEN) / (BLUE + RED + GREEN)
    bis = np.clip(bis, -1.5, 1.5)
    if verbose:
        mins = np.min(bis)
        maxs = np.max(bis)
        if mins < -1.5 or maxs > 1.5:
            print("bis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, bis[:, :, :, np.newaxis]], axis = -1)
    return x

def si(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    # (1 - B2) * (1 - B3) * (1 - B4) ** 1/3
    # This gets deleted before prediction, 
    BLUE = x[:, :, :, 0]
    RED = x[:, :, :, 2]
    GREEN = x[:, :, :, 1]
    sis = np.power( (1-BLUE) * (1 - GREEN) * (1 - RED), 1/3)
    if verbose:
        mins = np.min(sis)
        maxs = np.max(sis)
        if mins < -1 or maxs > 1:
            print("sis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, sis[:, :, :, np.newaxis]], axis = -1)
    return x

def ndmi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    ndmis = [(im[:, :, 5] - im[:, :, 9]) / (im[:, :, 5] + im[:, :, 9]) for im in x]
    ndmis = np.stack(ndmis)
    x = np.concatenate([x, ndmis[:, :, :, np.newaxis]], axis = -1)
    return x