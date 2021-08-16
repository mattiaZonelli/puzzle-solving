from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from sps.similarity import sim
from sps.psqp import compatibilities, psqp
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

        self.criterion = nn.TripletMarginLoss(config.get("margin", 1.))
        
        self.accuracy = Accuracy(num_classes=self.setup.num_tiles)
        self.device = config["device"]
        self.similarity = config["similarity"]
        self.verbose = config.get("verbose", False)

    def train(self, iterations, eval_step=1000):
        self.evaluate()
        trloss = 0.
        pbar = self._get_pbar(self.trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            loss = self.iteration(data)
            trloss += (loss.item() - trloss) / iter_

            if self.verbose:
                pbar.set_description(f"TRAIN - LOSS: {trloss:.4f}")

            if iter_ % eval_step == 0:
                self.evaluate()
                trloss = 0.

            if iter_ == iterations:
                break

    def _get_pbar(self, loader, desc):
        if self.verbose:
            return tqdm(loader, desc=desc)
        return loader

    def iteration(self, data):
        self.optimizer.zero_grad()

        position = data["position"]
        anchor = self.forward(x=data["anchor"],position=position)
        positive = self.forward(x=data["match"], position=-position)
        negative = self.forward(x=data["match"], position=position)

        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        with torch.no_grad():
            for data in pbar:
                puzzle = data["puzzle"].squeeze(0)
                emb_e = self.forward(x=puzzle, position=1)
                emb_w = self.forward(x=puzzle, position=-1)
                emb_s = self.forward(x=puzzle, position=2)
                emb_n = self.forward(x=puzzle, position=-2)

                Ch = sim(emb_e, emb_w, self.similarity)
                Cv = sim(emb_s, emb_n, self.similarity)

                A = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
                p = psqp(A, N=len(puzzle))
                dacc = self.accuracy(p, data["order"].squeeze()).item()
                
                if self.verbose:
                    pbar.set_description(f"VAL - DACC: {dacc:.4f}")

        self.model.train()
        self.accuracy.reset()

        return dacc

    def forward(self, **data):
        return self.model(**data)

    def reset(self, config):
        self.setup = self.setup_class(config)
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()


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
              "download": args.download,
              "data_dir": osp.abspath("./data/datasets"),
              "batch_size": args.batch_size,
              "lr": 1e-3 if args.lr is None else args.lr,
              "weight_decay": 0. if args.weight_decay is None else args.weight_decay,
              "momentum": 0.9 if args.momentum is None else args.momentum,
              "tile_size": args.tile_size,
              "similarity": args.similarity,
              "device": args.device,
              "device_ids": args.device_ids,
              "verbose": args.verbose,
              "savefig": args.savefig
              }

    trainer = Trainer(config)
    trainer.train(args.iterations)

