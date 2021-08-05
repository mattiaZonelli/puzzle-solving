import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, net):
        super(SiameseNet, self).__init__()
        self.net = net
        self.fcn = nn.Linear(self.net.fc.out_features + 1,
                             self.net.fc.out_features)

    def forward(self, anchor, positive, negative, position):
        anchor = self.net(anchor)
        positive = self.net(positive)
        negative = self.net(negative)

        position = torch.full((len(anchor),), position, device=anchor.device)

        anchor = self.fcn(torch.cat((anchor, position)))
        positive = self.fcn(torch.cat((positive, position)))
        negative = self.fcn(torch.cat((negative, position)))
        return anchor, positive, negative
