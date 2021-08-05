import random
from torch.utils.data import IterableDataset


class TileIterator:
    positions = [0, 1, 2, 3]  # 0: NORTH, 1: EAST, 2: SOUTH, 3: WEST

    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return self

    def _get_tile(self, img, x, y):
        return img[x:x + self.size, y:y + self.size]

    def __next__(self):
        size = self.size
        img = self.img_set[random.randrange(len(self.img_set))]
        _, w, h = img.shape

        # anchor
        xa, ya = random.randrange(w), random.randrange(h)
        anchor = self._get_tile(img, xa, ya)

        # positive
        xp, yp = None, None
        random.shuffle(self.positions)
        for pos in self.positions:
            if pos == "n" and ya - size >= 0:
                xp, yp = xa, ya - size
                break
            elif pos == "e" and xa + size <= w:
                xp, yp = xa + size, ya
                break
            elif pos == "s" and ya + size <= h:
                xp, yp = xa, ya + size
                break
            elif pos == "w" and xa - size >= 0:
                xp, yp = xa - size, ya
                break

        if xp is None or yp is None:
            raise ValueError("The puzzle tile size is too large."
                             "Cannot generate a positive pair.")

        positive = self._get_tile(img, xp, yp)

        # coordinates for total area of anchor + positive
        x0, y0 = min(xa, xp), min(ya, yp)
        xf, yf = max(xa + size, xp + size), max(ya + size, yp + size)

        # negative
        intx, inty = [], []
        if x0 - size >= 0:
            xn = intx.append(random.randrange(x0))
        if xf + size <= w:
            xn = random.choice([random.randrange(xf, w)] + intx)
        if y0 - size >= 0:
            yn = inty.append(random.randrange(y0))
        if y0 + size <= h:
            yn = random.choice([random.randrange(yf, h)] + inty)

        negative = self._get_tile(img, xn, yn)

        return {"anchor": anchor, "positive": positive, "negative": negative,
                "position": pos}


class TileDataset(IterableDataset):

    def __init__(self, img_set, size):
        self.img_set = img_set
        self.size = size

    def __iter__(self):
        return TileIterator(self.img_set, self.size)
