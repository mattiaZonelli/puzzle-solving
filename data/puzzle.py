import random
import os
import os.path as osp
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image
from skimage.util import view_as_windows
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (default_loader,
                                         has_file_allowed_extension, cast)
from torchvision.datasets.utils import download_and_extract_archive


def draw_puzzle(tiles, puzzle_size, separate=False):
    """ Draw the generated puzzle."""

    nh, nw = puzzle_size
    c, ts, _ = tiles.shape[1:]

    # nh, nw, ts, ts, c
    tiles = tiles.reshape(nh, nw, c, ts, ts).transpose(0, 1, 3, 4, 2)
    if separate:
        _, axes = plt.subplots(nh, nw)
        for i in range(nh):
            for j in range(nw):
                axes[i, j].imshow(tiles[i, j, :, :, :])
                axes[i, j].axis("off")
    else:
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(nh * ts, nw * ts, c)
        plt.imshow(tiles)
        plt.axis("off")
    plt.show()


class UnsupervisedImageFolder(ImageFolder):
    info = {"mit": ("http://people.csail.mit.edu/taegsang/Documents/jigsawCode.zip", osp.join("jigsawCode", "imData")),
            "mcgill": ("http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw-files/ImageDB-540parts-McGill.zip", ""),
            "bgu805": ("http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw-files/ImageDB-805parts-BGU.zip", ""),
            "bgu2360": ("http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw-files/ImageDB-2360parts-BGU.zip", ""),
            "bgu3300": ("http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw-files/ImageDB-3300parts-BGU.zip", "")}

    def __init__(self, name, root=None, transform=None, loader=default_loader,
                 is_valid_file=None, download=False):
        if download:
            download_url, extract_dir = self.info[name][0], osp.join(root, name)
            download_and_extract_archive(download_url, root, extract_dir)

        dset_root = osp.join(root, name, self.info[name][1])
        super(UnsupervisedImageFolder, self).__init__(dset_root, transform,
                                                      None, loader,
                                                      is_valid_file)
        del self.classes, self.class_to_idx  # Not needed

    def find_classes(self, directory):
        return [], {}

    @staticmethod
    def make_dataset(directory, class_to_idx=None, extensions=None,
                     is_valid_file=None):
        directory = osp.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x,
                                                  cast(Tuple[str, ...],
                                                       extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []

        for fname in sorted(os.listdir(directory)):
            path = os.path.join(directory, fname)
            if is_valid_file(path):
                item = path, None
                instances.append(item)

        if not len(instances):
            msg = f"Found no valid file in folder" + directory + ". "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, idx):
        return super(UnsupervisedImageFolder, self).__getitem__(idx)[0]


class PuzzleSet(UnsupervisedImageFolder):
    def __init__(self, name, root, tile_size, shuffle, transform=None,
                 loader=default_loader, is_valid_file=None, download=False):
        super(PuzzleSet, self).__init__(name, root, transform, loader,
                                        is_valid_file, download)
        self.tile_size = tile_size
        self.shuffle = shuffle

    def make_puzzle(self, img):
        size = self.tile_size

        if isinstance(img, Image):
            img = np.asarray(img).transpose(2, 0, 1)

        c, h, w = img.shape

        h, w = h // size * size, w // size * size
        img = img[:, :h, :w]
        tiles = view_as_windows(img, (c, size, size), size)
        _, nh, nw, _, _, _ = tiles.shape
        tiles = tiles.reshape(nh * nw, c, size, size)

        order = list(range(tiles.shape[0]))
        if self.shuffle:
            random.shuffle(order)
            tiles = tiles[order]
        return tiles, order, (nh, nw)

    def __getitem__(self, idx):
        img = super(PuzzleSet, self).__getitem__(idx)
        tiles, order, puzzle_size = self.make_puzzle(img)
        return {"puzzle": tiles, "order": order, "puzzle_size": puzzle_size,
                "tile_size": self.tile_size}
