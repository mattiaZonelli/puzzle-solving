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
        "--epochs",
        type=int,
        default=1,
        help="The number of training epochs."
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
        help="The batch size."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print a progressbar for training or not."
    )
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="*",
        default=None,
        help="The device id of the GPUS to be used during training."
    )
    return parser
