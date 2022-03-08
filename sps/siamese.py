import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, net):
        super(SiameseNet, self).__init__()
        self.net = net
        self.fcn = nn.Linear(self.net.fc.out_features + 1,
                             self.net.fc.out_features)

    def forward(self, x, position):
        x = self.net(x)

        if isinstance(position, int):
            position = torch.full((len(x), 1), position, device=x.device)

        x = self.fcn(torch.cat((x, position), dim=1))
        return x
