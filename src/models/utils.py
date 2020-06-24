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
    for x_offset, cval in enumerate([x for x in range(0, 140, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            base_id = 0
            subs = arr[:, cval:cval+16, rval:rval+16]
            images.append(subs)
            
    # Upright        
    for x_offset, cval in enumerate([x for x in range(7,  140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            base_id = 9*9
            subs = arr[:, cval:cval+16, rval:rval+16]
            images.append(subs)
            
    # Right
    for x_offset, cval in enumerate([x for x in range(7, 140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            base_id = (9*9)+(8*8)
            subs = arr[:, cval:cval+16, rval:rval+16]
            images.append(subs)

            
    # Up
    for x_offset, cval in enumerate([x for x in range(0, 140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            base_id = (9*9)+(8*8)+(9*8)
            subs = arr[:, cval:cval+16, rval:rval+16]
            images.append(subs)
    return images