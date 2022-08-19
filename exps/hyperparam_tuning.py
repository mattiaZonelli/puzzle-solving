import os
import os.path as osp
import random
from argparse import ArgumentParser
from functools import partial
from data import factory
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from exps.oracle import *
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser
from sps.siamese import SiameseNet
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm


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

    trload = DataLoader(trset, config["batch_size"], num_workers=1)
    vlload = DataLoader(vlset, 1, num_workers=1)
    return trload, vlload


def get_model():
    model = SiameseNet(resnet50(weights=ResNet50_Weights.DEFAULT))
    return model


def ray_train(config, checkpoint_dir="./"):
    model = get_model()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"],
                          momentum=config["momentum"])
    accuracyRot = Accuracy()

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trload, vlload = load_data(config)

    for epoch in range(10):
        # learning
        running_loss = 0.
        epoch_steps = 0
        pbar = _get_pbar(trload, "TRAIN")
        for iter_, data in enumerate(pbar, 1):
            loss = iteration(data, optimizer, criterion, model, device)  # for rotations
            running_loss += loss.item()
            epoch_steps += 1

            if VERBOSE:
                pbar.set_description(f"TRAIN - LOSS: {running_loss / epoch_steps:.4f}")
            if iter_ == ITERATIONS:
                '''running_loss = 0.
                epoch_steps = 0'''
                break

        # validation
        correct, total, dacc = validation(model, config, device) # for rotations

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            # print("\n....SAVING CHECKPOINT STATE in ", path, " ....\n")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune.report(train_loss=(running_loss / epoch_steps), val_loss=val_loss / val_steps)  # for hpo
        tune.report(loss=(running_loss / epoch_steps), accuracy=dacc)  # with accuracy
        # tune.report(loss=(running_loss / epoch_steps), accuracy=(correct / total))

        model.train()
        accuracyRot.reset()

    print("Finished Training")


def _get_pbar(loader, desc):
    if VERBOSE:
        return tqdm(loader, desc=desc)
    return loader


# iteration step for training on rotation
def iteration(data, optimizer, criterion, model, device):
    optimizer.zero_grad()

    target = torch.randint(4, (data['anchor'].shape[0],), device=device)
    tiles = data['anchor'].to(device)
    for i in range(data['anchor'].shape[0]):
        tiles[i] = torch.rot90(tiles[i], -target[i], [1, 2])

    output = model(x=tiles)

    before = list(model.parameters())

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # update weights

    after = model.parameters()
    assert before != after

    return loss


# validation for rotation
# return 1 only when we have perfect matching
def validation(model, config, device):
    accuracyRot = Accuracy()

    _, vlload = load_data(config)
    pbar = _get_pbar(vlload, "VAL")
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in pbar:
            puzzle = data["puzzle"].to(device)

            r = random.randrange(puzzle.shape[1])
            k = random.randrange(4)

            input = torch.rot90(puzzle[:, r, :, :], -k, [2, 3]).to(device)
            tmp = model(x=input)
            output = torch.zeros((tmp.shape[0], 4))
            output[:, torch.argmax(tmp)] = 1.
            target = torch.zeros((output.shape[0], 4))
            target[:, k] = 1.

            total += output.shape[0]
            correct += (output.squeeze().argmax() == k).sum().item()

            dacc = accuracyRot(output.squeeze(), target.int().squeeze())

            '''
            target = target.to(device)
            loss = criterion(tmp, target.argmax().unsqueeze(0))
            val_loss += loss.cpu().numpy()
            val_steps += 1'''

            if VERBOSE:
                pbar.set_description(f"VAL - loss: {dacc:.4f}")

    model.train()
    accuracyRot.reset()

    return correct, total, dacc


def train_parser():
    parser = ArgumentParser(parents=[train_abs_parser()])
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to plot and store explicative plots during the training."
    )
    return parser


VERBOSE = False
ITERATIONS = 450

if __name__ == "__main__":
    args = train_parser().parse_args()
    set_seed(args.seed)
    torch.cuda.empty_cache()

    model_config = {"dataset": args.dataset,
                    "download": args.download,
                    "data_dir": osp.abspath("./data/datasets"),
                    "batch_size": tune.choice([8, 16, 32, 64]),
                    "lr": tune.loguniform(1e-5, 1e-1),
                    "weight_decay": tune.loguniform(1e-10, 1e-3),
                    "momentum": tune.choice([0.8, 0.85, 0.9, 0.95]),
                    "tile_size": args.tile_size,
                    "similarity": args.similarity,
                    "device": args.device,
                    "device_ids": args.device_ids,
                    "verbose": args.verbose,
                    "savefig": args.savefig
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
        resources_per_trial={"cpu": 1, "gpu": 0.5},
        config=model_config,
        num_samples=5,  # number of trials for the hyperparameter optimization
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

    model_config["dataset"] = "mcgill"
    model_config["batch_size"] = 8
    print("---------------------------------------------------")

    best_trained_model = get_model()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint.pt"))
    best_trained_model.load_state_dict(model_state)

    correct, tot, test_acc = validation(best_trained_model, model_config, device=device)
    print("Best trial test set accuracy: {}".format(test_acc))
