import numpy as np
from math import sqrt

def slopePython(inBlock, outBlock, inXSize, inYSize, zScale=1):

    """ Calculate slope using Python.
        If Numba is available will make use of autojit function
        to run at ~ 1/2 the speed of the Fortran module.
        If not will fall back to pure Python - which will be slow!
    """
    for x in range(1,inBlock.shape[2]-1):
        for y in range(1, inBlock.shape[1]-1):
            # Get window size
            dx = 2 * inXSize[y,x]
            dy = 2 * inYSize[y,x]

            # Calculate difference in elevation
            dzx = (inBlock[0,y,x-1] - inBlock[0,y,x+1])*zScale
            dzy = (inBlock[0,y-1,x] - inBlock[0,y+1,x])*zScale

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0,y,x] = slopeDeg

    return outBlock

def slopePythonPlane(inBlock, outBlock, inXSize, inYSize, A_mat, z_vec, winSize=3, zScale=1):

    """ Calculate slope using Python.
        Algorithm fits plane to a window of data and calculated the slope
        from this - slope than the standard algorithm but can deal with
        noisy data batter.
        The matrix A_mat (winSize**2,3) and vector zScale (winSize**2) are allocated
        outside the function and passed in.
    """

    winOffset = int(winSize/2)

    for x in range(winOffset-1,inBlock.shape[2]):
        for y in range(winOffset-1, inBlock.shape[1]):
            # Get window size
            dx = winSize * inXSize[y,x]
            dy = winSize * inYSize[y,x]

            # Calculate difference in elevation
            """
                Solve A b = x to give x
                Where A is a matrix of:
                    x_pos | y_pos | 1
                and b is elevation
                and x are the coefficents
            """

            # Form matrix
            index = 0
            for i in range(-1*winOffset, winOffset+1):
                for j in range(-1*winOffset, winOffset+1):

                    A_mat[index,0] = 0+(i*inXSize[y,x])
                    A_mat[index,1] = 0+(j*inYSize[y,x])
                    A_mat[index,2] = 1

                    # Elevation
                    z_vec[index] = inBlock[0,y+j,x+i]*zScale

                    index+=1

            # Linear fit
            coeff_vec = np.linalg.lstsq(A_mat, z_vec)[0]

            # Calculate dzx and dzy
            dzx = coeff_vec[0] * dx
            dzy = coeff_vec[1] * dy

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0,y,x] = slopeDeg

    return outBlock


def calcSlope(inBlock, inXSize, inYSize, fitPlane=False, zScale=1, winSize=3, minSlope=None):
    """ Calculates slope for a block of data
        Arrays are provided giving the size for each pixel.
        * inBlock - In elevation
        * inXSize - Array of pixel sizes (x)
        * inYSize - Array of pixel sizes (y)
        * fitPlane - Calculate slope by fitting a plane to elevation
                     data using least squares fitting.
        * zScale - Scaling factor between horizontal and vertical
        * winSize - Window size to fit plane over.
    """
    # If fortran class could be imported use this
    # Otherwise run through loop in python (which will be slower)
    # Setup output block
    outBlock = np.zeros_like(inBlock, dtype=np.float32)
    if fitPlane:
        # Setup matrix and vector required for least squares fitting.
        winOffset = int(winSize/2)
        A_mat = np.zeros((winSize**2,3))
        z_vec = np.zeros(winSize**2)

        slopePythonPlane(inBlock, outBlock, inXSize, inYSize, A_mat, z_vec, zScale, winSize)
    else:
        slopePython(inBlock, outBlock, inXSize, inYSize, zScale)

    if minSlope is not None:
        # Set very low values to constant
        outBlock[0] = np.where(np.logical_and(outBlock[0] > 0,outBlock[0] < minSlope),minSlope,outBlock[0])
    return outBlock
