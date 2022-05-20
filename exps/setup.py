import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from data import factory
from sps.siamese import SiameseNet


class SiameseSetup:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        config = self.config
        trset, vlset = self.get_dsets()

        self.trload = DataLoader(trset, config["batch_size"], num_workers=1)
        self.vlload = DataLoader(vlset, 1, num_workers=1)
        return self.trload, self.vlload

    def get_dsets(self):
        config = self.config
        trset = factory(config["dataset"], size=config["tile_size"],
                        puzzle=False, root=config["data_dir"],
                        download=config.get("download", False))
        vlset = factory(config["dataset"], size=config["tile_size"],
                        puzzle=True, root=config["data_dir"],
                        download=config.get("download", False), shuffle=False)  # TODO: era shuffle:True
        self.num_tiles = len(vlset[0]["puzzle"])
        return trset, vlset

    def get_model(self):
        self.model = SiameseNet(resnet50(pretrained=True))
        return self.model

    def get_optimizer(self):
        config = self.config
        self.optimizer = optim.SGD(self.model.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])  # select which parameter to train
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)  # todo non so a che serve
        return self.optimizer  #, self.scheduler
