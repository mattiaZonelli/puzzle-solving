import torch
import torch.nn as nn
import torchvision.transforms as tt

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

        # for rotation_id5 and models train for rotation with only fc layer
        self.net.fc = nn.Linear(self.net.fc.in_features, 4)

        # for 'classification' (to load trained models on compatibility
        #self.net.fc = nn.Linear(self.net.fc.in_features, self.net.fc.out_features)
        # fcn is an extra layer on top of net.fc
        #self.fcn = nn.Linear(self.net.fc.out_features + 32, self.net.fc.out_features)

    # w/ SPP
    '''def forward(self, x):
        x = tt.Resize((224))(x)
        x = self.net(x).squeeze()
        x = x.permute(0, 3, 2, 1)

        m16 = nn.functional.max_pool2d(x, kernel_size=(1, x.shape[3] // 16), stride=(1, x.shape[3] // 16))
        m8 = nn.functional.max_pool2d(x, kernel_size=(1, x.shape[3] // 8), stride=(1, x.shape[3] // 8))
        m4 = nn.functional.max_pool2d(x, kernel_size=(1, x.shape[3] // 4), stride=(1, x.shape[3] // 4))
        m2 = nn.functional.max_pool2d(x, kernel_size=(1, x.shape[3] // 2), stride=(1, x.shape[3] // 2))
        m1 = nn.functional.max_pool2d(x, kernel_size=(1, x.shape[3] // 2), stride=(1, x.shape[3] // 1))
        m16 = m16.reshape(m16.shape[0], m16.shape[1]* m16.shape[2]* m16.shape[3])
        m8 = m8.reshape(m8.shape[0], m8.shape[1] * m8.shape[2] * m8.shape[3])
        m4 = m4.reshape(m4.shape[0], m4.shape[1] * m4.shape[2] * m4.shape[3])
        m2 = m2.reshape(m2.shape[0], m2.shape[1] * m2.shape[2] * m2.shape[3])
        m1 = m1.reshape(m1.shape[0], m1.shape[1] * m1.shape[2] * m1.shape[3])
        out = torch.cat((m16, m8, m4, m2, m1), dim=1)

        return out'''

    # for compatibility w/
    def forward(self, x, position):
        x = tt.Resize(168)(x)
        x = self.net(x).squeeze()

        if isinstance(position, int):
            position = torch.full((len(x), 1), position, device=x.device)
      
        x = self.fcn(torch.cat((x, position), dim=1))
        return x

    # for rotation
    '''def forward(self, x):
        x = self.net(x)
        return x'''
