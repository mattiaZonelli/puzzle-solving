import random
from torch.utils.data import IterableDataset


class TileIterator:
    # positions: 1: east, 2: south, -1: west, -2: north
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return self

    '''def _get_tile(self, img, x, y):
        if not (0 <= y <= img.shape[0] and 0. <= x <= img.shape[1]):
            raise ValueError(f"Piece size ({self.size}) is too large for "
                             f"retrieving tiles in image of size {img.shape}.")
        return img[y:y + self.size, x:x + self.size]'''

    def _get_tile(self, img, h, w):
        if not (0. <= h <= img.shape[1] and 0. <= w <= img.shape[2]):
            raise ValueError(f"Piece size ({self.size}) is too large for "
                             f"retrieving tiles in image of size {img.shape}.")
        if h + self.size >= img.shape[1]:
            print("\nsomething went wrong with the height!!\n")
        if w + self.size >= img.shape[2]:
            print("\nsomething went wrong with the width!!\n")
        '''
            sembra che se h+self.size o w+self.size sono maggiori della image.shape corrispondente, 
            vengano croppati.
            ma poi quando si fa h:h... o w:w... la "differenza" non + corretta
        '''
        out = img[0:img.shape[0], h:h + self.size, w:w + self.size]
        return out

    def __next__(self):
        size = self.size
        rimg = random.randrange(len(self.img_set))
        img = self.img_set[random.randrange(len(self.img_set))]
        _, h, w = img.shape

        # anchor
        # ha, wa = random.randrange(h - (size)), random.randrange(w - (size))  # next line was like this
        ha, wa = random.randrange(h - (size*2)), random.randrange(w - (size*2))
        anchor = self._get_tile(img, ha, wa, )

        # match ''' non ho ben capito a che serve '''
        pos = random.random() < 0.5  # False: HOR, True: VER
        hm, wm = ha + (size * (not pos)), wa + (size * pos),

        match = self._get_tile(img, hm, wm)

        return {"anchor": anchor, "match": match, "position": pos + 1}


class TileDataset(IterableDataset):
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return TileIterator(self.img_set, self.size)
