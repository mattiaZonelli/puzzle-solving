from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from sps.similarity import sim
from sps.psqp import compatibilities, psqp, psqp_ls
from exps.setup import SiameseSetup
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from data.puzzle import draw_puzzle
from exps.oracle import *

import random
from sps.relab import solve_puzzle
import matplotlib.pyplot as plt


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
                # return trloss

    def _get_pbar(self, loader, desc):
        if self.verbose:
            return tqdm(loader, desc=desc)
        return loader

    def iteration(self, data):
        self.optimizer.zero_grad()
        '''
            se effettivamente le due dimensioni di data["anchor“] e data["match"] corrispondono, 
            questo dovrebbe funzionare
        '''
        position = data["position"]
        position = torch.ones(1, len(data["anchor"])) * position
        position = position.t()

        anchor = self.forward(x=data["anchor"], position=position)
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

                '''
                emb_e = self.forward(x=puzzle, position=1)

                emb_w = self.forward(x=puzzle, position=-1)

                emb_s = self.forward(x=puzzle, position=2)

                emb_n = self.forward(x=puzzle, position=-2)

                Ch = sim(emb_e, emb_w, self.similarity)
                Cv = sim(emb_s, emb_n, self.similarity)
                '''

                '''Ch, Cv = oracle_compatibilities_og(data)
                A = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
                # A_dense = A.to_dense()
                p = psqp_ls(A, N=len(puzzle))
                dacc = self.accuracy(p.squeeze(), data["order"].squeeze()).item()'''
                h = 2
                r = 2
                lim_h = 8
                lim_r = 8
                psqp_dacc = torch.zeros(lim_r - r + lim_h - h)
                rl_dacc = torch.zeros(lim_r - r + lim_h - h)
                n_tiles = torch.zeros(lim_r - r + lim_h - h)
                i = 1
                while h < lim_h and r < lim_r:
                    order = list(range(h * r))
                    random.shuffle(order)
                    order = torch.tensor([order]).int()
                    # relab = solve_puzzle((h, r), order).int()
                    for k in range(20):
                        relab = solve_puzzle((h, r), order).int()
                        try:
                            t_dacc = self.accuracy(relab.squeeze(), order.squeeze()).item()
                        except:
                            t_dacc = my_accuracy(relab.squeeze(), order.squeeze(), h * r)
                        if t_dacc > rl_dacc[i]:
                            rl_dacc[i] = t_dacc

                    data['puzzle_size'] = torch.tensor([h, r]).unsqueeze(0)
                    Ch, Cv = oracle_compatibilities(h, r, order)
                    A = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
                    p = psqp_ls(A, N=(h * r))
                    try:
                        psqp_dacc[i] = self.accuracy(p.squeeze(), order.squeeze()).item()
                    except:
                        psqp_dacc[i] = my_accuracy(p.squeeze(), order.squeeze(), h * r)
                    if psqp_dacc[i] < 1e-3:
                        for k in range(15):
                            p = psqp_ls(A, N=(h * r))
                            dacc = my_accuracy(p.squeeze(), order.squeeze(), h * r)
                            if dacc > psqp_dacc[i]:
                                psqp_dacc[i] = dacc

                    n_tiles[i] = h * r
                    if h == r:
                        r += 1
                    else:
                        h += 1
                    i += 1

                fileName = r'n_tile vs accuracy.png'
                fig, ax = plt.subplots(1)
                plt.plot(n_tiles, rl_dacc, label='ReLab Accuracy')
                plt.plot(n_tiles, psqp_dacc, label='PSQP Accuracy')
                plt.legend()
                plt.xlabel('# tiles')
                plt.ylabel('Accuracy')
                plt.show()
                fig.savefig(fileName, format='png')
                plt.close(fig)

                if self.verbose:
                    pbar.set_description(f"VAL - DACC: {psqp_dacc[i-1]:.4f}")

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
              "savefig": args.savefig,
              }

    trainer = Trainer(config)
    trainer.train(args.iterations, args.eval_step)

    '''
    start_time = time.time()
    loss = 0.
    for it in range(10):
        loss += trainer.train(args.iterations)

    print("AVG LOSS: ", loss/10)
    print("--- %s seconds ---" % (time.time() - start_time))
    '''

'''
    NOTE:
    - flag --savefig sembra non funzionare;
    - non sembra venire croppato;
    - --tile_size 252 -> 2x2
    - --tile_size 168 -> 3x4
    - --tile_size 126 -> 4x5
'''
