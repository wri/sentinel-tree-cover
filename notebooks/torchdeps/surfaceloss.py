import torch
import numpy as np
from torch import Tensor, einsum

from utils import simplex, probs2one_hot, one_hot
from utils import one_hot2hd_dist


import argparse
from pathlib import Path
from operator import add
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import partial

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import torch.sparse
from tqdm import tqdm
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt as eucl_distance

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss