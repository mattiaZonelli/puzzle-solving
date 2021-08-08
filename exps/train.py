from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from exps.setup import SiameseSetup
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser


class Trainer:
    setup_class = SiameseSetup

    def __init__(self, config):
        self.setup = self.setup_class(config)

        self.trload, self.vlload = self.setup.get_loaders()
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()

        self.criterion = nn.TripletMarginLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", False)
        self.accuracy = Accuracy(num_classes=self.setup.n_classes + 1,
                                 ignore_index=self.ignore_index,
                                 compute_on_step=False).to(self.device)
        self.curr_epoch = 0

    def train(self, epochs):
        self.evaluate()
        for epoch in range(epochs):
            self.epoch()
            self.evaluate()

    def _get_pbar(self, loader, desc):
        if self.verbose:
            return tqdm(loader, total=len(self.trload), desc=desc)
        return loader

    def epoch(self):
        mloss = 0.

        pbar = self._get_pbar(self.trload, "TRAIN")
        self.model.train()
        for n, data in enumerate(pbar, 1):
            self.optimizer.zero_grad()

            anchor, positive, negative = self.forward(data)

            loss = self.criterion(anchor, positive, negative)
            loss.backward()
            self.optimizer.step()
            mloss += (loss.item() - mloss) / n
            if self.verbose:
                pbar.set_description(f"TRAIN - LOSS: {mloss:.4f}")
        self.curr_epoch += 1

        return mloss

    def evaluate(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        mloss = 0.

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        with torch.no_grad():
            for n, data in enumerate(pbar, 1):
                anchor, positive, negative = self.forward(data)
                loss = self.criterion(anchor, positive, negative)
                mloss += (loss.item() - mloss) / n

                if self.verbose:
                    pbar.set_description(f"VAL - LOSS: {mloss:.4f}, ")

        self.accuracy.reset()

        return mloss

    def forward(self, data):
        anchor = data["anchor"].to(self.device)
        positive = data["positive"].to(self.device)
        negative = data["negative"].to(self.device)
        return self.model(anchor, positive, negative, data["position"])

    def reset(self, config):
        self.setup = self.setup_class(config)
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()
        self.curr_epoch = 0


def train_parser():
    parser = ArgumentParser(parents=[train_abs_parser()])
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to plot and store explicative plots during the training."
    )
    return parser


if __name__ == "__main__":
    args = train_parser().parse_args()
    set_seed(args.seed)

    config = {"dataset": args.dataset,
              "data_dir": osp.abspath("./datasets"),
              "batch_size": args.batch_size,
              "lr": 1e-3 if args.lr is None else args.lr,
              "weight_decay": 0. if args.weight_decay is None else args.weight_decay,
              "momentum": 0.9 if args.momentum is None else args.momentum,
              "device": "cuda",
              "device_ids": args.device_ids,
              "verbose": args.verbose,
              "savefig": args.savefig
              }
    trainer = Trainer(config)
    trainer.train(args.epochs)

