import numpy as np

def tile_images(arr: np.ndarray) -> list:
    """ Converts a 632x632 array to a 289, 16, 16 array
        
        Parameters:
         arr (np.ndaray): (632, 632) float array
    
        Returns:
         images (list): 
    """

    # Normal
    images = []
    for x_offset, cval in enumerate([x for x in range(1, 140, 14)]):
        for y_offset, rval in enumerate([x for x in range(1, 140, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 142])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 142])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 4), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (4, 0), (0, 0), (0, 0)), 'reflect')
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (4, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 4), (0, 0)), 'reflect')
            print(subs.shape)
            images.append(subs)
            
    # Upright        
    for x_offset, cval in enumerate([x for x in range(8,  142-8, 14)]):
        for y_offset, rval in enumerate([x for x in range(8, 142-8, 14)]):
            base_id = 9*9
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 142])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 142])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 4), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (4, 0), (0, 0), (0, 0)), 'reflect')
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (4, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 4), (0, 0)), 'reflect')
            print(subs.shape)
            images.append(subs)
            
    # Right
    for x_offset, cval in enumerate([x for x in range(8, 142-8, 14)]):
        for y_offset, rval in enumerate([x for x in range(1, 140, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 142])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 142])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 4), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (4, 0), (0, 0), (0, 0)), 'reflect')
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (4, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 4), (0, 0)), 'reflect')
            print(subs.shape)
            images.append(subs)

            
    # Up
    for x_offset, cval in enumerate([x for x in range(1, 142-8, 14)]):
        for y_offset, rval in enumerate([x for x in range(8, 142-8, 14)]):
            min_x = np.max([cval - 5, 0])
            max_x = np.min([cval + 19, 142])
            min_y = np.max([rval - 5, 0])
            max_y = np.min([rval + 19, 142])
            subs = arr[:, min_x:max_x, min_y:max_y]
            if x_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 4), (0, 0), (0, 0)), 'reflect')
            if x_offset == 9:
                subs = np.pad(subs, ((0, 0), (4, 0), (0, 0), (0, 0)), 'reflect')
            if y_offset == 0:
                subs = np.pad(subs, ((0, 0), (0, 0), (4, 0), (0, 0)), 'reflect')
            if y_offset == 9:
                subs = np.pad(subs, ((0, 0), (0, 0), (0, 4), (0, 0)), 'reflect')
            print(subs.shape)
            images.append(subs)
    return images