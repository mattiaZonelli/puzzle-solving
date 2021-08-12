import random
from torch.utils.data import IterableDataset


class TileIterator:
    # positions: 1: east, 2: south, -1: west, -2: north
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return self

    def _get_tile(self, img, x, y):
        if not (0 <= y <= img.shape[0] and 0. <= x <= img.shape[1]):
            raise ValueError(f"Piece size ({self.size}) is too large for "
                             f"retrieving tiles in image of size {img.shape}.")
        return img[y:y + self.size, x:x + self.size]

    def __next__(self):
        size = self.size
        img = self.img_set[random.randrange(len(self.img_set))]
        _, h, w = img.shape

        # anchor
        xa, ya = random.randrange(w - size), random.randrange(h - size)
        anchor = self._get_tile(img, xa, ya)

        # match
        pos = random.random() < 0.5  # False: HOR, True: VER
        xm, ym = xa + (size * pos), ya + (size * (not pos))

        match = self._get_tile(img, xm, ym)
        return {"anchor": anchor, "match": match, "position": pos + 1}


class TileDataset(IterableDataset):
    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return TileIterator(self.img_set, self.size)
