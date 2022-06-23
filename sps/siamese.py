import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, net):
        super(SiameseNet, self).__init__()
        self.net = net  # ResNet50

        '''print('freeze conv layers')
        for param in self.parameters():  # for transfer learning, ResNet AS fixed ftrs extractor
            param.requires_grad = False
            
        # to freeze also batchnNorm layers
        for name, child in (self.net.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                child.eval()
        print('frozen batchNorm2d')'''

        print('finetuning')

        # for rotation attempt 1
        '''lin = self.net.fc
        new_lin = nn.Sequential(
            lin,
            nn.Linear(self.net.fc.out_features, 4) #  is an extra layer on top of net.fc
        )
        self.net.fc = new_lin'''
        # for rotation_id5
        self.net.fc = nn.Linear(self.net.fc.in_features, 4)

        # for 'classification'
        '''self.net.fc = nn.Linear(self.net.fc.in_features, self.net.fc.out_features)
        # fcn is an extra layer on top of net.fc
        self.fcn = nn.Linear(self.net.fc.out_features + 1,
                             self.net.fc.out_features)'''

    '''def forward(self, x, position):
        x = self.net(x).squeeze()

        if isinstance(position, int):
            position = torch.full((len(x), 1), position, device=x.device)

        x = self.fcn(torch.cat((x, position), dim=1))
        return x'''


    # for rotation_id5
    def forward(self, x):
        x = self.net(x)
        return x
