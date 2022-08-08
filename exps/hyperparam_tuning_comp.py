import os
import os.path as osp
import random
from argparse import ArgumentParser
from functools import partial

import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from data import factory
from exps.oracle import *
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from sps.siamese import SiameseNet
from sps.similarity import sim, myaccuracy


def get_dsets(config):
    trset = factory("mit", size=config["tile_size"],
                    puzzle=False, root=config["data_dir"],
                    download=config.get("download", False))
    vlset = factory("mcgill", size=config["tile_size"],
                    puzzle=True, root=config["data_dir"],
                    download=config.get("download", False), shuffle=False)  # TODO: era shuffle:True

    return trset, vlset


def load_data(config):
    trset, vlset = get_dsets(config)
    num_tiles = len(vlset[0]["puzzle"])

    trload = DataLoader(trset, config["batch_size"], num_workers=1)
    vlload = DataLoader(vlset, 1, num_workers=1)
    return trload, vlload, num_tiles


def get_model_from(device, current_dir=None):
    model = SiameseNet(resnet50(weights=ResNet50_Weights.DEFAULT))

    if current_dir is None:
        path_to_model = os.path.join(os.path.abspath("../../../"), OG_MODEL)
    else:
        path_to_model = os.path.join(os.path.abspath("./"), OG_MODEL)

    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
    #modello1.model.eval()
    # azzera / cancella il precedente fully connected layer che era per la rotazione
    model.net.fc = nn.Identity()
    # nuovo fully connected layer da allenare per la compatibilità
    model.fcn = nn.Linear(2048 + 1, 1000)
    return model


def ray_train(config, checkpoint_dir="./"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    model = get_model_from(device)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"],
                          momentum=config["momentum"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trload, vlload, num_tiles = load_data(config)

    criterion = nn.TripletMarginLoss(config.get("margin", 1.))
    #accuracy = Accuracy(num_classes=num_tiles)

    for epoch in range(10):
        # learning
        running_loss = 0.
        epoch_steps = 0
        pbar = _get_pbar(trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            loss = iteration(data, optimizer, criterion, model, device=device)  # for
            running_loss += loss.item()
            epoch_steps += 1

            if VERBOSE:
                pbar.set_description(f"TRAIN - LOSS: {running_loss / epoch_steps:.4f}")
            if iter_ == ITERATIONS:
                '''running_loss = 0.
                epoch_steps = 0'''
                break

        # validation
        correct, total, dacc = validation(model, config, device) # for

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            # print("\n....SAVING CHECKPOINT STATE in ", path, " ....\n")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune.report(train_loss=(running_loss / epoch_steps), val_loss=val_loss / val_steps)  # for hpo
        tune.report(loss=(running_loss / epoch_steps), accuracy=dacc)  # with accuracy
        # tune.report(loss=(running_loss / epoch_steps), accuracy=(correct / total))

        model.train()
        #accuracy.reset()

    print("Finished Training")


def _get_pbar(loader, desc):
    if VERBOSE:
        return tqdm(loader, desc=desc)
    return loader


# iteration step for training on
def iteration(data, optimizer, criterion, model, device):
    optimizer.zero_grad()

    position = data['position'].unsqueeze(1).to(device)
    '''in base alla position divido l'anchor in hor/ver e il match nell'opposto'''
    n_tiles, channels, h, w = data['anchor'].shape
    anchor_in = torch.empty((n_tiles, channels, h, w // 2), device=device)
    pos_in = torch.empty((n_tiles, channels, h, w // 2), device=device)
    neg_in = torch.empty((n_tiles, channels, h, w // 2), device=device)
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

    anchor = model(x=anchor_in, position=position)
    positive = model(x=pos_in, position=-position)
    negative = model(x=neg_in, position=-position)

    before = list(model.fcn.parameters())

    loss = criterion(anchor, positive, negative)
    loss.backward()
    optimizer.step()  # update weights

    after = model.fcn.parameters()
    assert before != after

    return loss


# validation
def validation(model, config, device):
    similarity = config["similarity"]

    trload, vlload, num_tiles = load_data(config)

    accuracy = Accuracy(num_classes=num_tiles)
    pbar = _get_pbar(vlload, "VAL")
    model.eval()
    val_acc = torch.zeros(20)
    pbar_n = 0
    with torch.no_grad():
        for data in pbar:
            puzzle = data["puzzle"].squeeze(0).to(device)

            # to eval comp of model trained for rotation w/out last fc
            '''emb_e = self.forward(x=puzzle)
            emb_w = self.forward(x=puzzle)
            emb_s = self.forward(x=puzzle)
            emb_n = self.forward(x=puzzle)'''

            # evaluate comp of model trained w/ fc for comp
            emb_e = model(x=puzzle, position=1)
            emb_w = model(x=puzzle, position=-1)
            emb_n = model(x=puzzle, position=2)
            emb_s = model(x=puzzle, position=-2)

            Ch = sim(emb_e, emb_w, similarity)
            Cv = sim(emb_s, emb_n, similarity)
            Ch[Ch > 0.] = 1.
            Cv[Cv > 0.] = 1.

            Ch2, Cv2 = oracle_compatibilities_og(data)
            # dacc = self.accuracy(Ch.argmax(dim=1), Ch2.argmax(dim=1)).item()
            # dacc2 = self.accuracyRot(Cv.argmax(dim=1), Cv2.argmax(dim=1)).item()

            dacc = myaccuracy(Ch, Ch2)
            dacc2 = myaccuracy(Cv, Cv2)
            val_acc[pbar_n] = (dacc + dacc2) / 2

            if VERBOSE:
                    pbar.set_description(f"VAL - DACC > Ch: {dacc:.4f}, Cv: {dacc2:.4f}")
            pbar_n += 1

            # draw_puzzle(data['puzzle'].squeeze(), (3, 4), separate=True)

    print(f"VAL - MEAN DACC: {torch.mean(val_acc):.4f}")
    model.train()
    accuracy.reset()

    return dacc, dacc2, torch.mean(val_acc)


VERBOSE = False
ITERATIONS = 450
OG_MODEL = "ray_rot00.pt"

if __name__ == "__main__":

    args = ArgumentParser(parents=[train_abs_parser()]).parse_args()
    set_seed(args.seed)
    torch.cuda.empty_cache()

    model_config = {"dataset": args.dataset,
                    "download": args.download,
                    "data_dir": osp.abspath("./data/datasets"),
                    "batch_size": tune.choice([8, 16, 32]),
                    "lr": tune.loguniform(1e-5, 1e-1),
                    "weight_decay": tune.loguniform(1e-10, 1e-3),
                    "momentum": tune.choice([0.8, 0.85, 0.9, 0.95]),
                    "tile_size": args.tile_size,
                    "similarity": args.similarity,
                    "device": args.device,
                    "device_ids": args.device_ids,
                    "verbose": args.verbose
                    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        time_attr="training_iteration",
        max_t=args.iterations,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(ray_train),
        local_dir="./ray_results",
        resources_per_trial={"cpu": 1, "gpu": 0.5},
        config=model_config,
        num_samples=10,  # number of trials for the hyperparameter optimization
        scheduler=scheduler,
        progress_reporter=reporter,
        stop={"training_iteration": 1},
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # validate on testset
    model_config["dataset"] = "mcgill"
    model_config["batch_size"] = 8
    print("---------------------------------------------------")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model = get_model_from(device, current_dir=os.getcwd())
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint.pt"))
    best_trained_model.load_state_dict(model_state)

    correct, tot, test_acc = validation(best_trained_model, model_config, device=device)
    print("Best trial test set accuracy: {}".format(test_acc))
