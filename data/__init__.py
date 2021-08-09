import os.path as osp

from data.puzzle import UnsupervisedImageFolder, PuzzleSet
from data.tiles import TileDataset


def factory(name="mit", puzzle=False, root=None, download=False, shuffle=False,
            size=None):
    """

    :param name: one of "mit", "mcgill", "bgu805", "bgu2360", "bgu3300"
    :param root: the root where the data is stored.
    :param download: whether or not to download the data.
    :return: PuzzleSet: the puzzle set.
    """
    if puzzle and size is None:
        raise ValueError("'size' must not be None when 'puzzle' is True.")

    if root is None:
        root = osp.join(".", "data", "datasets")

    if puzzle:
        return PuzzleSet(name, root, size, shuffle, download=download))
    return TileDataset(UnsupervisedImageFolder(name, root, download=download))
