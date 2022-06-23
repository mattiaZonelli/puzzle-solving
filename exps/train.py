import torch
from tqdm import tqdm
from argparse import ArgumentParser
import os.path as osp
import torch.nn as nn
from torchmetrics import Accuracy
from sps.similarity import *
from sps.psqp import compatibilities, psqp_ls
from exps.setup import SiameseSetup
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from exps.oracle import *
import random
from sps.relab import solve_puzzle
import matplotlib.pyplot as plt
from tests.validate_state_dicts import validate_state_dicts
from data.puzzle import draw_puzzle

PATH = "./rotation6.pt"


class Trainer:
    setup_class = SiameseSetup

    def __init__(self, config):
        self.setup = self.setup_class(config)

        self.trload, self.vlload = self.setup.get_loaders()
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()

        # self.criterion = nn.TripletMarginLoss(config.get("margin", 1.))
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(num_classes=self.setup.num_tiles)
        self.accuracyRot = Accuracy()
        self.device = config["device"]
        self.similarity = config["similarity"]
        self.verbose = config.get("verbose", False)
        self.model.to(self.device)  # TODO to set the model to CUDA
        print(self.device)

    def train(self, iterations, eval_step=1000):
        # self.evaluate()
        # self.validation()
        tr_losses = torch.ones(iterations + 1)
        # val_losses = torch.ones(iterations + 1)
        trloss = 0.
        # vlloss = 0.

        pbar = self._get_pbar(self.trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            # loss = self.iteration_compatibility(data)  # for compatibilities
            loss = self.iteration(data)  # for rotations
            trloss += (loss.item() - trloss) / iter_
            tr_losses[iter_] = trloss

            if self.verbose:
                pbar.set_description(f"TRAIN - LOSS: {trloss:.4f}")

            if iter_ > 1 and tr_losses[iter_] > tr_losses[iter_ - 1]:
                torch.save(self.model.state_dict(), PATH)

            if iter_ % eval_step == 0:
                # if iter_ <= iterations:
                self.evaluate()
                '''loss1 = self.validation()
                vlloss += (loss1.item() - vlloss) / iter_
                val_losses[iter_] = vlloss'''
                # trloss = 0.  # PERCHE' VIENE AZZERATA AD OGNI VALIDATION??

            if iter_ == iterations:
                fileName = r'train_val_loss.png'
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

        # k = random.randrange(4)  # 0 = 0°, 1 = 90°, 2=180°, 3 = 270°
        target = torch.randint(4, (data['anchor'].shape[0],), device=self.device)
        tiles = data['anchor'].to(self.device)
        for i in range(data['anchor'].shape[0]):
            tiles[i] = torch.rot90(tiles[i], -target[i], [1, 2])
            '''plt.imshow(tiles[i].permute(1, 2, 0))
            plt.axis("off")
            plt.show()'''
        output = self.forward(x=tiles)

        before = list(self.model.parameters())

        # target = torch.full((output.shape[0],), fill_value=k, device=self.device)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()  # update weights

        after = self.model.parameters()
        assert before != after

        return loss

    def iteration_compatibility(self, data):
        self.optimizer.zero_grad()

        position = data['position'].unsqueeze(1).to(self.device)
        '''in base alla position divido l'anchor in hor/ver e il match nell'opposto'''
        n_tiles, channels, h, w = data['anchor'].shape
        anchor_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        pos_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        neg_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        for i in range(position.shape[0]):
            tile_size = h
            if position[i] == 1:  # HOR
                data['match'][i] = torch.rot90(data['match'][i], 1, [1, 2])
            else:  # VER
                data['anchor'][i] = torch.rot90(data['anchor'][i], 1, [1, 2])

            anchor_in[i] = data['anchor'][i, :, :, :tile_size // 2]
            pos_in[i] = data['anchor'][i, :, :, tile_size // 2:tile_size]
            up_down = random.random() < 0.5
            if up_down == 0:
                neg_in[i] = data['match'][i, :, :, :tile_size // 2]
            else:
                neg_in[i] = data['match'][i, :, :, tile_size // 2:tile_size]

        '''anchor_in = data['anchor'].to(self.device)
        pos_in = data['match'].to(self.device)
        neg_in = pos_in'''

        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)

        before = list(self.model.fcn.parameters())

        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        after = self.model.fcn.parameters()
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
                puzzle = data["puzzle"].squeeze(0).to(self.device)

                '''emb_e = self.forward(x=puzzle)
                emb_w = self.forward(x=puzzle)
                emb_s = self.forward(x=puzzle)
                emb_n = self.forward(x=puzzle)'''

                emb_e = self.forward(x=puzzle, position=1)
                emb_w = self.forward(x=puzzle, position=-1)
                emb_n = self.forward(x=puzzle, position=2)
                emb_s = self.forward(x=puzzle, position=-2)

                Ch = sim(emb_e, emb_w, self.similarity)
                Cv = sim(emb_s, emb_n, self.similarity)

                '''Ch = psqp_Ch(puzzle)
                Ch[Ch > 0.2] = 1.
                Ch[Ch <= 0.2] = 0.'''

                Ch[Ch > 0.] = 1.
                Cv[Cv > 0.] = 1.

                Ch2, Cv2 = oracle_compatibilities_og(data)
                # dacc = self.accuracy(Ch.argmax(dim=1), Ch2.argmax(dim=1)).item()
                # dacc2 = self.accuracyRot(Cv.argmax(dim=1), Cv2.argmax(dim=1)).item()
                dacc = myaccuracy(Ch, Ch2)
                dacc2 = myaccuracy(Cv, Cv2)
                val_acc[pbar.n] = (dacc + dacc2) / 2

                if self.verbose:
                    if dacc < 0.5 or dacc2 < 0.5:
                        puzzle_list = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9]
                        pbar.set_description(f"VAL - DACC > Ch: {dacc:.4f}, Cv: {dacc2:.4f}")
                        print("PUZZLE N° ", puzzle_list[pbar.n], '\n')
                        draw_puzzle(data['puzzle'].squeeze(), data['puzzle_size'].squeeze())
                    else:
                        pbar.set_description(f"VAL - DACC > Ch: {dacc:.4f}, Cv: {dacc2:.4f}")

                '''A2 = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
                pi = psqp_ls(A2, N=len(puzzle))
                dacc = self.accuracy(pi.squeeze(), data["order"].squeeze()).item()

                if self.verbose:
                    pbar.set_description(f"VAL - DACC > {dacc:.4f}")'''
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

                r = random.randrange(puzzle.shape[1])
                k = random.randrange(4)

                input = torch.rot90(puzzle[:, r, :, :], -k, [2, 3])
                tmp = self.forward(x=input)
                output = torch.zeros(tmp.shape[0], 4)
                output[:, torch.argmax(tmp)] = 1.
                target = torch.zeros(output.shape[0], 4)
                target[:, k] = 1.

                dacc = self.accuracyRot(output.squeeze(), target.int().squeeze())
                val_loss += dacc

                '''tmp_loss = self.criterion(output, torch.tensor(k))
                val_loss += tmp_loss'''
                if self.verbose:
                    pbar.set_description(f"VAL - dacc: {dacc:.4f}")

        self.model.train()
        self.accuracyRot.reset()
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
    torch.cuda.empty_cache()

    config = {"dataset": args.dataset,
              "download": args.download,
              "data_dir": osp.abspath("./data/datasets"),
              "batch_size": args.batch_size,
              "lr": 1e-5 if args.lr is None else args.lr,
              "weight_decay": 0. if args.weight_decay is None else args.weight_decay,
              "momentum": 0.9 if args.momentum is None else args.momentum,
              "tile_size": args.tile_size,
              "similarity": args.similarity,
              "device": args.device,
              "device_ids": args.device_ids,
              "verbose": args.verbose,
              "savefig": args.savefig,
              }

    '''trainer = Trainer(config)
    trainer.train(args.iterations, args.eval_step)
    torch.save(trainer.model.state_dict(), PATH)'''

    #modello1 = Trainer(config)
    # training
    '''modello1.model.load_state_dict(torch.load("./rotation_id5.pt", map_location=torch.device(modello1.device)))
    modello1.model.eval()
    # azzera / cancella il precedente fully connected layer che era per la rotazione
    modello1.model.net.fc = nn.Identity()
    # nuovo fully connected layer da allenare per la compatibilità
    modello1.model.fcn = nn.Linear(2048 + 1, 1000)
    modello1.optimizer = modello1.setup.get_optimizer()
    modello1.model.to(modello1.device)
    modello1.train(args.iterations, args.eval_step)
    torch.save(modello1.model.state_dict(), PATH)'''

    # validation on compatibility - on CPU
    '''modello1.model.net.fc = nn.Identity()
    modello1.model.fcn = nn.Linear(2048 + 1, 1000)
    modello1.model.load_state_dict(torch.load("./half-tiles_frozen.pt", map_location=torch.device(modello1.device)))
    modello1.model.to(modello1.device)
    modello1.model.eval()
    modello1.evaluate()'''


    config["dataset"] = 'mit'
    modello1 = Trainer(config)
    modello1.model.load_state_dict(torch.load("./rotation_id5.pt", map_location=torch.device(modello1.device)))
    modello1.model.eval()
    mit_dacc = torch.zeros(20)
    for i in range(20):
        mit_dacc[i] = modello1.validation()

    config["dataset"] = 'mcgill'
    modello2 = Trainer(config)
    modello2.model.load_state_dict(torch.load("./rotation_id5.pt", map_location=torch.device(modello2.device)))
    modello2.model.eval()
    mcgill_dacc = torch.zeros(20)
    for i in range(20):
        mcgill_dacc[i] = modello2.validation()

    fileName = r'mit-mcgill_rotation5.png'
    fig, ax = plt.subplots(1)
    plt.plot(torch.tensor(range(20)), mit_dacc, label='MIT Accuracy')
    plt.plot(torch.tensor(range(20)), mcgill_dacc, label='McGill Accuracy')
    plt.legend()
    plt.xlabel('# iterations')
    plt.ylabel('Accuracy')
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)

    # validate_state_dicts(model_state_dict_1=modello1.model.state_dict(), model_state_dict_2=modello2.model.state_dict())
