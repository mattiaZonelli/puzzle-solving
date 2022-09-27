import copy
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import IterableDataset


class TileIterator:
    # positions: 1: east, -1: west, 2: north, -2: south
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return self

    def _get_tile(self, img, h, w):
        if not (0. <= h <= img.shape[1] and 0. <= w <= img.shape[2]):
            raise ValueError(f"Piece size ({self.size}) is too large for "
                             f"retrieving tiles in image of size {img.shape}.")
        if h + self.size - 1 >= img.shape[1]:
            print(f"\nsomething went wrong with the height!! - {h + self.size - 1}\n")
        if w + self.size - 1 >= img.shape[2]:
            print(f"\nsomething went wrong with the width!! - {w + self.size - 1}\n")
        '''
            sembra che se h+self.size o w+self.size sono maggiori della image.shape corrispondente, 
            vengano croppati.
            ma poi quando si fa h:h... o w:w... la "differenza" non più corretta
        '''
        out = img[0:img.shape[0], h:h + self.size, w:w + self.size]
        return out

    # next tile for horizontal compatibility network
    def __next__(self):
        size = self.size
        img = self.img_set[random.randrange(len(self.img_set))]
        _, h, w = img.shape

        ht = h // size
        wt = w // size

        # ROTATION
        # ha, wa = size * random.randrange(ht), size * random.randrange(wt)  # "sequenziale"
        # HOR
        ha, wa = size * random.randrange(ht), size * random.randrange(wt - 1)  # "sequenziale"
        # ha, wa = random.randrange(h - size), random.randrange(w - (size * 2))  # random
        # VER
        # ha, wa = size * random.randrange(ht-1), size * random.randrange(wt)  # "sequenziale"
        # ha, wa = random.randrange(h - (size  *2)), random.randrange(w - size)  # random
        anchor = self._get_tile(img, ha, wa)
        pos = False  # False: HOR, True: VER
        hm, wm = ha + (size * pos), wa + (size * (not pos))
        # hm, wm = ha, wa  # ROTATION
        match = self._get_tile(img, hm, wm)

        return {"anchor": anchor, "match": match, "position": pos + 1}


class TileDataset(IterableDataset):
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return TileIterator(self.img_set, self.size)
