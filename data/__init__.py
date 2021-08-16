import os.path as osp
from torchvision.transforms import ToTensor

from data.puzzle import UnsupervisedImageFolder, PuzzleSet
from data.tiles import TileDataset


def factory(name="mit", size=28, puzzle=False, root=None, download=False,
            shuffle=False):
    """

    :param name: one of "mit", "mcgill", "bgu805", "bgu2360", "bgu3300"
    :param root: the root where the data is stored.
    :param download: whether or not to download the data.
    :param shuffle: whether to shuffle or not the puzzle.
    :param size: the tile size.
    :return: Union[PuzzleSet, TileDataset]: the puzzle set or the tile dataset.
    """
    if puzzle and size is None:
        raise ValueError("'size' must not be None when 'puzzle' is True.")

    if root is None:
        root = osp.join(".", "data", "datasets")

    if puzzle:
        return PuzzleSet(name, root, size, shuffle, transform=ToTensor(), download=download)
    return TileDataset(UnsupervisedImageFolder(name, root,
                                               transform=ToTensor(),
                                               download=download), size=size)
