import os.path as osp
from argparse import ArgumentParser


def input_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=osp.join("..", "experiment"),
        help="The experiment folder."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mit",
        help="The dataset."
    )
    parser.add_argument(
        "--similarity",
        type=str,
        default="gaussian",  # todo "gaussian"
        help="The similarity function."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Whether or not to download the dataset."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device (cpu or cuda)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed for the experiment."
    )
    return parser


def train_abs_parser():
    parser = ArgumentParser(parents=[input_parser()], add_help=False)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="The number of training iterations."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="The learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="The weight decay."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="The momentum."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size for the training set."
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=10,
        required=True,
        help="The size of a (square) puzzle tile."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print a progressbar for training or not."
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=10,
        help="The number of iterations after which the evaluation is run."
    )
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="*",
        default=None,
        help="The device id of the GPUS to be used during training."
    )
    return parser
