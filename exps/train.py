import torch
from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch.nn as nn
from torchmetrics import Accuracy
from sps.siamese import SiameseNet
from sps.similarity import *
from sps.psqp import compatibilities, psqp_ls
from exps.setup import SiameseSetup
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from data.puzzle import draw_puzzle
from exps.oracle import *
import random
from sps.relab import solve_puzzle
import matplotlib.pyplot as plt
from tests.validate_state_dicts import validate_state_dicts

PATH = "./snn_oracle.pt"

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
        #self.evaluate()
        #self.validation()
        tr_losses = torch.ones(iterations + 1)
        #val_losses = torch.ones(iterations + 1)
        trloss = 0.
        #vlloss = 0.

        pbar = self._get_pbar(self.trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            loss = self.iteration(data)
            trloss += (loss.item() - trloss) / iter_
            tr_losses[iter_] = trloss

            if self.verbose:
                pbar.set_description(f"TRAIN - LOSS: {trloss:.4f}")

            if iter_ > 1 and tr_losses[iter_] > tr_losses[iter_-1]:
                PATH = "./snn_oracle.pt"
                torch.save(self.model.state_dict(), PATH)

            if iter_ % eval_step == 0:
            # if iter_ <= iterations:
                self.evaluate()
                '''loss1 = self.validation()
                vlloss += (loss1.item() - vlloss) / iter_
                val_losses[iter_] = vlloss'''
                # trloss = 0.  # PERCHE' VIENE AZZERATA AD OGNI VALIDATION??

            if iter_ == iterations:
                fileName = r'Train vs Validation losses.png'
                fig, ax = plt.subplots(1)
                plt.plot(range(iterations), tr_losses[range(1, iterations + 1)], label='Train Loss')
                # plt.plot(range(iterations), val_losses[range(1, iterations + 1)], label='Validation Loss')
                plt.legend()
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.show()
                fig.savefig(fileName, format='png')
                plt.close(fig)
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
        '''position = data["position"]
        position = torch.ones(1, len(data["anchor"])) * position
        position = position.t()'''
        position = data['position'].unsqueeze(1)

        anchor = self.forward(x=data["anchor"], position=position)
        positive = self.forward(x=data["match"], position=-position)
        negative = self.forward(x=data["match"], position=position)

        before = list(self.model.parameters())

        loss = self.criterion(anchor, positive, negative)
        loss.backward()

        self.optimizer.step()  # update weights

        after = self.model.parameters()

        assert before != after

        return loss

    def evaluate(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        val_acc = torch.zeros(20)
        with torch.no_grad():
            for data in pbar:
                puzzle = data["puzzle"].squeeze(0)

                emb_e = self.forward(x=puzzle, position=1)

                emb_w = self.forward(x=puzzle, position=-1)

                emb_s = self.forward(x=puzzle, position=-2)

                emb_n = self.forward(x=puzzle, position=2)

                Ch = sim(emb_e, emb_w, self.similarity)
                Cv = sim(emb_s, emb_n, self.similarity)

                '''Ch = psqp_Ch(puzzle)
                Ch[Ch > 0.2] = 1.
                Ch[Ch <= 0.2] = 0.'''

                Ch[Ch > 0.] = 1.
                Cv[Cv > 0.] = 1.

                Ch2, Cv2 = oracle_compatibilities_og(data)
                A = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
                pi = psqp_ls(A, N=len(puzzle))
                if torch.min(pi) < 0.:
                    print(pi)
                dacc = self.accuracy(pi.squeeze(), data["order"].squeeze()).item()
                '''h = 2
                r = 2
                lim_h = 9
                lim_r = 9
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
                plt.close(fig)'''
                val_acc[pbar.n] = dacc
                if self.verbose:
                    pbar.set_description(f"VAL - DACC: {dacc:.4f}")
        print(f"VAL - MEAN DACC: {torch.mean(val_acc):.4f}")
        self.model.train()
        self.accuracy.reset()

        return dacc

    def validation(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        val_loss = 0.
        with torch.no_grad():
            for data in pbar:
                puzzle = data["puzzle"]

                pos = random.random() < 0.5
                pos += 1
                if pos == 1:
                    r = random.randrange(12)
                    if (r + 1) % 4 == 0:
                        r -= 1
                    r1 = r+1
                elif pos == 2:
                    r = random.randrange(8)
                    r1 = r+4

                emb_e = self.forward(x=puzzle[:, r, :, :], position=pos)

                emb_w = self.forward(x=puzzle[:, r1, :, :], position=-pos)

                neg = self.forward(x=puzzle[:, r1, :, :], position=pos)

                tmp_loss = self.criterion(emb_e, emb_w, neg)
                val_loss += tmp_loss

                if self.verbose:
                    pbar.set_description(f"VAL - LOSS: {tmp_loss:.4f}")

        self.model.train()
        self.accuracy.reset()
        return val_loss / 20

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

    torch.save(trainer.model.state_dict(), PATH)

    '''modello1 = Trainer(config)
    modello1.model.load_state_dict(torch.load("./snn_oracle_batch32it850.pt"))
    modello1.model.eval()'''
    #siamese.evaluate()
    '''modello2 = Trainer(config)
    modello2.model.load_state_dict(torch.load("./snn_finetune_batch32it850.pt"))
    modello2.model.eval()
    modello2.evaluate()'''

    #validate_state_dicts(model_state_dict_1=modello1.model.state_dict(), model_state_dict_2=modello2.model.state_dict())


''' 
    Qual'è il senso di avere fc e poi fcn?
    
    batch32it550:   gaussian top3 -> Ch 7-8/12  , CV 1-2/12
                    correlation top1 -> Ch 2/12 , Cv none
                    correlation top3 -> Ch 9/12 , Cv none
    batch512it100:  correlation top1 -> Ch 3/12 other ok stessa riga, Cv none
                    correlation top3 -> Ch 8-9/12, Cv 1/12
                    gaussian top1 -> Ch 3/12    , Cv 3/12
                    gaussian top3 -> Ch 9-10/12 , CV none
    batch32it850:   gaussian top3 -> Ch 9-10/12 , CV 2/12
                    correlation top3 -> Ch 8-9/12, Cv 1/12
                    
    

'''
