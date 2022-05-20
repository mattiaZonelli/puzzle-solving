import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, net):
        super(SiameseNet, self).__init__()
        self.net = net  # ResNet50
        '''for param in self.parameters():  # for transfer learning, Resnet AS fixed ftrs extractor
            param.requires_grad = False'''

        self.net.fc = nn.Linear(self.net.fc.in_features, self.net.fc.out_features)
        # fcn is an extra layer on top of net.fc
        self.fcn = nn.Linear(self.net.fc.out_features + 1,
                             self.net.fc.out_features)  # non sono convinto sulle out_ftrs

    def forward(self, x, position):
        x = self.net(x)

        if isinstance(position, int):
            position = torch.full((len(x), 1), position, device=x.device)

        x = self.fcn(torch.cat((x, position), dim=1))
        return x


'''
    fc : out ftrs = 2048
    fcn : in ftrs = 2048+1,  out ftrs = 1000 ??? 
    
    Che output voglio dall mi modello? Qualcosa per calcolare le compatibilities
'''