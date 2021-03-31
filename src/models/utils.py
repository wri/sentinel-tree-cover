import numpy as np

def tile_images(arr: np.ndarray) -> list:
    """ Converts a 142x142 array to a 289, 24, 24 array
        
        Parameters:
         arr (np.ndaray): (142, 142) float array
    
        Returns:
         images (list): 
    """

    # Normal
    images = []
    for x_offset, cval in enumerate([x for x in range(0, 140, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 140])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 140])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (5, 0), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 5), (0, 0), (0, 0)), 'reflect')
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (5, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 5), (0, 0)), 'reflect')
            images.append(subs)
            
    # Upright  
    for x_offset, cval in enumerate([x for x in range(7,  140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            base_id = 9*9
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 140])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 140])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 5), (0, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 5), (0, 0)), 'reflect')
            if (subs.shape[2] != 24) or (subs.shape[1] != 24):
                print(subs.shape, min_x, max_x, min_y, max_y, x_offset, "UPRIGHT")
            images.append(subs)
    """
    # Right
    for x_offset, cval in enumerate([x for x in range(7, 140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 140])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 140])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (5, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 5), (0, 0)), 'reflect')
            if (subs.shape[2] != 24) or (subs.shape[1] != 24):
                print(subs.shape, x_offset, y_offset, "RIGHT")
            images.append(subs)

            
    # Up
    for x_offset, cval in enumerate([x for x in range(0, 140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 140])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 140])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (5, 0), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 5), (0, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (5, 0), (0, 0)), 'reflect')
            if (subs.shape[2] != 24) or (subs.shape[1] != 24):
                print(subs.shape, x_offset, y_offset, "UP")
            images.append(subs)
    """
    return images