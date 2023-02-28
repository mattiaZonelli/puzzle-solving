import os.path as osp
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser
from torchmetrics import Accuracy
from exps.setup import SiameseSetup
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from exps.oracle import *
from sps.compatibility import *
from sps.similarity import *
from sps.psqp import *
from sps.relab import solve_puzzle
from sps.post_processing import *
from tests.validate_state_dicts import validate_state_dicts
from data.puzzle import draw_puzzle

PATH = "./zzztest.pt"


# loaded_pos = torch.load("float_pos.txt")


class Trainer:
    setup_class = SiameseSetup

    def __init__(self, config):
        self.setup = self.setup_class(config)

        self.trload, self.vlload = self.setup.get_loaders()
        self.model = self.setup.get_model()
        self.optimizer = self.setup.get_optimizer()

        self.criterion = nn.TripletMarginLoss(config.get("margin", 1.))
        # self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(num_classes=self.setup.num_tiles)
        self.accuracyRot = Accuracy()
        self.device = config["device"]
        self.similarity = config["similarity"]
        self.verbose = config.get("verbose", False)
        self.model.to(self.device)  # to set the model to CUDA
        print(self.device)

    def train(self, iterations, eval_step=1000):
        # self.evaluate()
        # self.validation()
        tr_losses = torch.ones(iterations + 1)
        val_losses = torch.ones(iterations // eval_step)
        trloss = 0.

        running_loss = 0.
        epochs = 0
        vlloss = 0.
        vli = 0

        pbar = self._get_pbar(self.trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            loss = self.iteration_compatibility(data)  # for compatibility
            # loss = self.iteration(data)  # for rotations
            trloss += (loss.item() - trloss) / iter_
            tr_losses[iter_] = trloss

            running_loss += loss.item()
            epochs += 1

            if self.verbose:
                pbar.set_description(f"TRAIN - LOSS: {running_loss / epochs:.4f}")

            if iter_ > 1 and tr_losses[iter_] > tr_losses[iter_ - 1]:
                torch.save(self.model.state_dict(), PATH)

            if iter_ % 500 == 0:
                print("\nIteration: ", iter_, f" - \nTRAIN - LOSS: {running_loss / epochs:.4f}")

            if iter_ % eval_step == 0:
                _, vlloss = self.evaluate()
                # _, vlloss = self.validation()
                val_losses[vli] = vlloss
                vli += 1

            if iter_ == iterations:
                fileName = r'zCOMP_mit28-tloss.png'
                fig, ax = plt.subplots(1)
                plt.plot(range(iterations), tr_losses[range(1, iterations + 1)], label='Train Loss')
                # plt.plot(range(0, iterations-1, eval_step), val_losses, label='Validation Loss')
                # plt.plot(range(iterations), val_losses[range(1, iterations + 1)], label='Validation Loss')
                plt.legend()
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.show()
                fig.savefig(fileName, format='png')
                plt.close(fig)
                print(f"\nTRAIN - LOSS: {running_loss / epochs:.4f}")
                break
                # return trloss

    def _get_pbar(self, loader, desc):
        if self.verbose:
            return tqdm(loader, desc=desc)
        return loader

    # iteration step for training on rotation
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

    # iteration step on training for compatibility
    def iteration_compatibility(self, data):
        self.optimizer.zero_grad()

        position = data['position'].unsqueeze(1).to(self.device)
        # in base alla position divido l'anchor in hor/ver e il match nell'opposto
        n_tiles, channels, h, w = data['anchor'].shape
        anchor_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        pos_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        neg_in = torch.empty((n_tiles, channels, h, w // 2), device=self.device)
        # HALF TILES - OG
        '''for i in range(n_tiles):
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
                neg_in[i] = data['match'][i, :, :, tile_size // 2:tile_size]'''

        # HALF TILES  - TEN inspired
        for i in range(n_tiles):
            tile_size = h
            anchor_in[i] = data['anchor'][i, :, :, tile_size // 2:tile_size]
            pos_in[i] = data['match'][i, :, :, :tile_size // 2]
            neg_in[i] = data['match'][i, :, :, tile_size // 2:tile_size]

        # FULL TILES - metto in pos e neg la tiles a destra o sotto in base alla position
        '''n_tiles, channels, h, w = data['anchor'].shape
        anchor_in = data['anchor'].to(self.device)
        pos_in = torch.empty((n_tiles, channels, h, w), device=self.device)
        neg_in = torch.empty((n_tiles, channels, h, w), device=self.device)
        for i in range(position.shape[0]):
            if position[i] == 1:  # HOR
                pos_in[i] = data['right'][i]
                neg_in[i] = data['down'][i]
            else:  # VER
                pos_in[i] = data['down'][i]
                neg_in[i] = data['right'][i]'''

        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)

        '''anchor = self.forward(x=anchor_in)
        pos_in = torch.rot90(pos_in, 2, [2, 3])
        positive = self.forward(x=pos_in)
        negative = self.forward(x=neg_in)'''

        before = list(self.model.parameters())

        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        # for NAP_
        for i in range(n_tiles):
            tile_size = h
            neg_in[i] = data['anchor'][i, :, :, :tile_size // 2]
            anchor_in[i] = data['anchor'][i, :, :, tile_size // 2:tile_size]
            pos_in[i] = data['match'][i, :, :, :tile_size // 2]

        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)
        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        # for _NAP
        for i in range(n_tiles):
            tile_size = h
            neg_in[i] = data['anchor'][i, :, :, tile_size // 2:tile_size]
            anchor_in[i] = data['match'][i, :, :, :tile_size // 2]
            pos_in[i] = data['match'][i, :, :, tile_size // 2:tile_size]

        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)
        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        after = self.model.parameters()
        assert before != after

        return loss

    def iteration_compatibility_VER(self, data):
        self.optimizer.zero_grad()

        position = data['position'].unsqueeze(1).to(self.device)

        n_tiles, channels, h, w = data['anchor'].shape
        anchor_in = torch.empty((n_tiles, channels, h // 2, w), device=self.device)
        pos_in = torch.empty((n_tiles, channels, h // 2, w), device=self.device)
        neg_in = torch.empty((n_tiles, channels, h // 2, w), device=self.device)

        # HALF TILES  - TEN inspired
        for i in range(n_tiles):
            tile_size = h
            anchor_in[i] = data['anchor'][i, :, tile_size // 2:tile_size, :]
            pos_in[i] = data['match'][i, :, :tile_size // 2, :]
            neg_in[i] = data['match'][i, :, tile_size // 2:tile_size, :]
        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)

        before = list(self.model.parameters())

        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        # for NAP_
        for i in range(n_tiles):
            tile_size = h
            neg_in[i] = data['anchor'][i, :, :tile_size // 2, :]
            anchor_in[i] = data['anchor'][i, :, tile_size // 2:tile_size, :]
            pos_in[i] = data['match'][i, :, :tile_size // 2, :]
        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)
        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        # for _NAP
        for i in range(n_tiles):
            tile_size = h
            neg_in[i] = data['anchor'][i, :, tile_size // 2:tile_size, :]
            anchor_in[i] = data['match'][i, :, :tile_size // 2, :]
            pos_in[i] = data['match'][i, :, tile_size // 2:tile_size, :]
        anchor = self.forward(x=anchor_in, position=position)
        positive = self.forward(x=pos_in, position=-position)
        negative = self.forward(x=neg_in, position=-position)
        loss = self.criterion(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()  # update weights

        after = self.model.parameters()
        assert before != after

        return loss

    # function to compare relab and psqp on puzzles with different sizes w/ oracle compatibility
    def compare(self, vlload=None):

        self.model.eval()

        # puzzle = data["puzzle"].squeeze(0).to(self.device)
        # minimum sizes of puzzle
        h = 2
        r = 2
        # maximum sizes of puzzles
        lim_h = 20
        lim_r = 20
        psqp_dacc = torch.zeros(lim_r - r + lim_h - h)
        rl_dacc = torch.zeros(lim_r - r + lim_h - h)
        n_tiles = torch.zeros(lim_r - r + lim_h - h)
        i = 1
        while h < lim_h and r < lim_r:
            print("r is at: ", r, " | h is at: ", h, " \n")
            order = list(range(h * r))
            random.shuffle(order)
            order = torch.tensor([order]).int()
            Ch, Cv = oracle_compatibilities(h, r, order)
            A = compatibilities(Ch, Cv, torch.tensor([h, r]))

            # relab = solve_puzzle((h, r), order).int()
            for k in range(1):
                # relab = solve_puzzle((h, r), order).int()
                relab, _ = solve_puzzle((h, r), A)
                try:
                    t_dacc = self.accuracy(relab.squeeze(), order.squeeze()).item()
                except:
                    t_dacc = puzzle_accuracy(relab.squeeze(), order.squeeze(), h * r)
                if t_dacc > rl_dacc[i]:
                    rl_dacc[i] = t_dacc

            p, _ = psqp_ls(A, N=(h * r))
            try:
                psqp_dacc[i] = self.accuracy(p.squeeze(), order.squeeze()).item()
            except:
                psqp_dacc[i] = puzzle_accuracy(p.squeeze(), order.squeeze(), h * r)
            if psqp_dacc[i] < 1e-3:
                for k in range(15):
                    _, p = psqp_ls(A, N=(h * r))
                    dacc = puzzle_accuracy(p.squeeze(), order.squeeze(), h * r)
                    if dacc > psqp_dacc[i]:
                        psqp_dacc[i] = dacc

            n_tiles[i] = h * r
            if h == r:
                r += 1
            else:
                h += 1
            i += 1

        fileName = r'relab_psqp_oracle_20x20.png'
        fig, ax = plt.subplots(1)
        plt.plot(n_tiles, rl_dacc, label='ReLab Accuracy')
        plt.plot(n_tiles, psqp_dacc, label='PSQP Accuracy')
        plt.legend()
        plt.xlabel('# tiles')
        plt.ylabel('Accuracy')
        plt.show()
        fig.savefig(fileName, format='png')
        plt.close(fig)
        print("FINE")

    # validation function for compatibility + solving puzzle
    def evaluate(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        val_acc = torch.zeros(20)
        val_acc2 = torch.zeros(20)
        with torch.no_grad():
            for data in pbar:
                puzzle = data["puzzle"].squeeze(0).to(self.device)

                # to eval comp of model trained for rotation w/out last fc
                '''emb_w = self.forward(x=puzzle)
                emb_e = self.forward(x=puzzle)
                emb_n = self.forward(x=puzzle)
                emb_s = self.forward(x=puzzle)'''

                '''emb_w = self.forward(x=puzzle)
                emb_e = self.forward(x=torch.rot90(puzzle, -2, (2, 3)))
                emb_n = self.forward(x=torch.rot90(puzzle, 1, (2, 3)))
                emb_s = self.forward(x=torch.rot90(puzzle, -1, (2, 3)))'''

                # w/ position
                emb_w = self.forward(x=puzzle, position=1)
                emb_e = self.forward(x=puzzle, position=-1)
                # for i in range(puzzle.shape[0]):
                #    puzzle[i] = torch.rot90(puzzle[i], 1, [1, 2])
                emb_n = self.forward(x=puzzle, position=2)
                emb_s = self.forward(x=puzzle, position=-2)
                # emb_n = vertical_ext(x=puzzle, position=2)
                # emb_s = vertical_ext(x=puzzle, position=-2)

                Ch = sim(emb_w, emb_e, self.similarity)
                Cv = sim(emb_n, emb_s, self.similarity)

                '''Ch = psqp_Ch(puzzle)
                Ch[Ch > 0.2] = 1.
                Ch[Ch <= 0.2] = 0.'''

                Ch[Ch > 0.] = 1.
                Cv[Cv > 0.] = 1.

                # Similarity by Gallagher mixed Cho et al.
                '''try:
                    Ch7 = mgc(x=data['puzzle'].squeeze())
                    Ch7[Ch7 > 1e-6] = 1.
                    Ch7[Ch7 < 1e-6] = 0.
                except:
                    Ch7 = compatibility_Ch(data['puzzle'].squeeze())
                    Ch7[Ch7 > 0.01] = 1.
                    Ch7[Ch7 < 0.01] = 0.
                # Ch7 = compatibility_Ch(data['puzzle'].squeeze())
                # Ch7[Ch7 > 0.01] = 1.
                # Ch7[Ch7 < 0.01] = 0.
                Cv7 = compatibility_Cv(data['puzzle'].squeeze())
                Cv7[Cv7 > 0.01] = 1.
                Cv7[Cv7 < 0.01] = 0.'''

                Ch_oracle, Cv_oracle = oracle_compatibilities_og(data)
                # dacc = self.accuracy(Ch.argmax(dim=1), Ch_oracle.argmax(dim=1)).item()
                # dacc2 = self.accuracyRot(Cv.argmax(dim=1), Cv_oracle.argmax(dim=1)).item()

                #
                c_tiles = constant_tiles(data["puzzle"].squeeze()).nonzero()
                #

                dacc = myaccuracy(Ch, Ch_oracle, c_tiles)
                dacc2 = myaccuracy(Cv, Cv_oracle, c_tiles)
                # dacc_h7 = self.accuracy(Ch7.argmax(dim=1), Ch_oracle.argmax(dim=1)).item()
                # dacc_v7 = myaccuracy(Cv7, Cv_oracle)
                val_acc[pbar.n] = dacc
                val_acc2[pbar.n] = dacc2
                # val_acc2[pbar.n] = dacc_h7

                if self.verbose:
                    #   pbar.set_description(f"VAL - DACC > Ch: {dacc:.4f}, Cv: {dacc2:.4f} | Comp. by Gallagher mixed Pomeranz {dacc_h7:.4f}, {dacc_v7:.4f}")
                    #   print("")
                    pbar.set_description(f"VAL - DACC > Ch: {dacc:.4f}, Cv: {dacc2:.4f}")

                # for puzzle solving
                # devo azzerare le compatibilità degli altri e verso gli altri per tutte le constant tiles
                '''for c in c_tiles:
                    Ch[c, :] = 0.
                    Ch[:, c] = 0.
                    Cv[c, :] = 0.
                    Cv[:, c] = 0.

                A = compatibilities(Ch, Cv_oracle, data["puzzle_size"].squeeze())
                pi, p = psqp_ls(A, N=len(puzzle))
                # pi, p = solve_puzzle((data['puzzle_size'][0, 0], data['puzzle_size'][0, 1]), A)
                puzzle_ACC = self.accuracy(pi, data["order"].squeeze()).item()

                # POST-PROCESSING
                pi, p = missing_tiles2(pi, p, puzzle_ACC, c_tiles, data['order'].squeeze())
                puzzle_ACC = self.accuracy(pi, data["order"].squeeze()).item()
                pi, p = missing_tiles3(pi, p, puzzle_ACC, c_tiles, data['order'].squeeze())
                # pi, p = missing_tiles(p.reshape(pi.shape[0], pi.shape[0]), A, pi, data['order'].squeeze())

                pi, p = cyclical_shift(puzzle_ACC, p, data['puzzle_size'].squeeze(), data['order'].squeeze())
                puzzle_ACC2 = self.accuracy(pi, data["order"].squeeze()).item()

                # puzzle_ACC2 = puzzle_accuracy(pi, data['order'].squeeze(), 12, c_tiles)
               
                val_acc[pbar.n] = puzzle_ACC2

                draw_puzzle(puzzle[pi.to(torch.int64)], (data['puzzle_size'][0, 0], data['puzzle_size'][0, 1]),
                            separate=False)

                if self.verbose:
                    pbar.set_description(f"VAL - DACC > {puzzle_ACC2:.4f}")'''

        print(f"VAL - MEAN DACC: {torch.mean(val_acc):.4f} - {torch.mean(val_acc2):.4f}")
        self.model.train()
        self.accuracy.reset()

        return torch.mean(val_acc)


    # validation for rotation
    def validation(self, vlload=None):
        if vlload is None:
            vlload = self.vlload

        pbar = self._get_pbar(vlload, "VAL")
        self.model.eval()
        val_acc = 0.
        loss = 0.

        with torch.no_grad():
            for data in pbar:
                puzzle = data["puzzle"]

                r = random.randrange(puzzle.shape[1])
                k = random.randrange(4)

                input = torch.rot90(puzzle[:, r, :, :], -k, [2, 3]).to(self.device)
                tmp = self.forward(x=input)
                output = torch.zeros(tmp.shape[0], 4)
                output[:, torch.argmax(tmp)] = 1.
                target = torch.zeros(output.shape[0], 4)
                target[:, k] = 1.

                dacc = self.accuracyRot(output.squeeze(), target.int().squeeze())
                val_acc += dacc

                if self.verbose:
                    pbar.set_description(f"VAL - dacc: {dacc:.4f}")

                loss += self.criterion(tmp, torch.tensor(k, device=self.device).unsqueeze(0)).item()

            self.model.train()
            self.accuracyRot.reset()
            return val_acc / 20  # , loss / 20

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
    '''start_time = time.time()
    trainer = Trainer(config)
    # per riprendere il training di un vecchio modello
    #trainer.model.load_state_dict(torch.load("./ray_rot02_mcgill.pt", map_location=torch.device(trainer.device)))
    trainer.train(args.iterations, args.eval_step)
    torch.save(trainer.model.state_dict(), PATH)
    print("--- %s seconds ---" % (time.time() - start_time))'''

    modello1 = Trainer(config)
    # training on compatibility
    '''start_time = time.time()
    modello1.model.load_state_dict(torch.load("./zROT_mit28.pt", map_location=torch.device(modello1.device)))
    # modello1.model.eval()
    # modello1.model.net = nn.Sequential(*list(modello1.model.net.children())[:-2]
    # nn.Conv2d(2048, 2048, (1, 1)),
    # nn.Conv2d(2048, 2048, (1, 1))
    #                                   )
    # azzera / cancella il precedente fully connected layer che era per la rotazione
    modello1.model.net.fc = nn.Identity()
    # modello1.model.net = nn.Sequential(*list(modello1.model.net.children())[:-2])
    # nuovo fully connected layer da allenare per la compatibilità
    modello1.model.fcn = nn.Linear(2048 + 1, 1000)
    modello1.optimizer = modello1.setup.get_optimizer()
    modello1.model.to(modello1.device)
    modello1.train(args.iterations, args.eval_step)
    torch.save(modello1.model.state_dict(), PATH)
    print("--- %s seconds ---" % (time.time() - start_time))'''

    # validation on compatibility - on CPU - use correlation!
    '''modello1.model.net.fc = nn.Identity()
    modello1.model.fcn = nn.Linear(2048 + 1, 1000)
    # modello1.model.net = nn.Sequential(*list(modello1.model.net.children())[:-2]
    # nn.Conv2d(2048, 2048, (1, 1)),
    # nn.Conv2d(2048, 2048, (1, 1))
    #
    # modello1.model.load_state_dict(torch.load("./ten_VERv3NAPx2.pt", map_location=torch.device(modello1.device)))
    modello1.model.load_state_dict(torch.load("./ten_HORv2NAPx2.pt", map_location=torch.device(modello1.device)))
    # modello1.model.net.fc = nn.Identity()  # for model trained on rotation
    modello1.model.to(modello1.device)
    modello1.model.eval()
    # modello1.evaluate()
    modello1.mix_puzzle()'''

    # ver_ext = Trainer(config)
    # ver_ext.model.net.fc = nn.Identity()
    # ver_ext.model.fcn = nn.Linear(2048 + 1, 1000)
    # ver_ext.model.load_state_dict(torch.load("./ten_VERv3NAPx2.pt", map_location=torch.device(ver_ext.device)))
    # ver_ext.model.to(ver_ext.device)
    # ver_ext.model.eval()
    # modello1.evaluate(ver_ext.model)

    # validation on rotation
    '''config["dataset"] = 'mit'
    modello1 = Trainer(config)
    modello1.model.load_state_dict(torch.load("./zROT_mit28.pt", map_location=torch.device(modello1.device)))
    modello1.model.eval()
    mit_dacc = torch.zeros(20)
    for i in range(20):
        mit_dacc[i] = modello1.validation()

    config["dataset"] = 'mcgill'
    modello2 = Trainer(config)
    modello2.model.load_state_dict(torch.load("./zROT_mit28.pt", map_location=torch.device(modello2.device)))
    modello2.model.eval()
    mcgill_dacc = torch.zeros(20)
    for i in range(20):
        mcgill_dacc[i] = modello2.validation()

    fileName = r'zROT28-validation.png'
    fig, ax = plt.subplots(1)
    plt.plot(torch.tensor(range(20)), mit_dacc, label='MIT Accuracy')
    plt.plot(torch.tensor(range(20)), mcgill_dacc, label='McGill Accuracy')
    plt.legend()
    plt.xlabel('# iterations')
    plt.ylabel('Accuracy')
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)'''

# validate_state_dicts(model_state_dict_1=modello1.model.state_dict(), model_state_dict_2=modello2.model.state_dict())
