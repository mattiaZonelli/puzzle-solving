from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from exps.setup import SiameseSetup


class Trainer:
    setup_class = SiameseSetup

    def __init__(self, config):
        self.setup = self.setup_class(config)

        self.trload, self.vlload = self.setup.get_loaders()
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()

        self.ignore_index = self.setup.ignore_index
        self.criterion = nn.TripletMarginLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", False)
        self.savefig = config.get("savefig", False)
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

                if self.savefig:
                    self.callback()
        self.accuracy.reset()

        return mloss

    def callback(self, **kwargs):
        pass

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

    exp_dir = osp.join("results", args.exp_dir)
    comps_dir = osp.join(exp_dir, "compatibilities")

    base_fname = f"ks={args.kernel_size}_dil={args.dilation}"
    pt_fname = base_fname + ".pt"

    config = {"dataset": "pascalvoc",
              "data_dir": osp.abspath("./datasets"),
              "batch_size": args.batch_size,
              # "pt_fname": osp.abspath(osp.join(comps_dir, pt_fname)),
              "pt_fname": osp.abspath(exp_dir),
              "rl_params": {"kernel_size": args.kernel_size,
                            "dilation": args.dilation,
                            "iterations": args.iterations},
              "lr": args.lr or 1e-3,
              "weight_decay": args.weight_decay or 0.,
              "momentum": args.momentum or 0.9,
              "device": "cuda",
              "device_ids": args.device_ids,
              "verbose": args.verbose,
              "savefig": args.savefig
              }
    trainer = DeepLabReLabTrainer(config)
    trainer.train(args.epochs)

