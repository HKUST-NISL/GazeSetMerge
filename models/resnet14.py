from typing import Optional, Tuple, Union

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


from .attention import Attention

class ResNet(torchvision.models.ResNet):
    def __init__(self):
        block_name = 'basic'
        if block_name == 'basic':
            block = torchvision.models.resnet.BasicBlock
        elif block_name == 'bottleneck':
            block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError
        layers = [2, 2, 2, 1]
        super().__init__(block, layers)
        # del self.layer4
        # del self.avgpool
        # del self.fc

        pretrained_name = 'resnet18'
        if pretrained_name:
            state_dict = torchvision.models.utils.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)

            print('Load pretrained from %s' % (torchvision.models.resnet.model_urls[pretrained_name]))



    def forward(self, x: torch.tensor) -> torch.tensor:

        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out.append(x)

        x = self.layer1(x)
        out.append(x)

        x = self.layer2(x)
        out.append(x)

        x = self.layer3(x)
        out.append(x)

        x = self.layer4(x)
        out.append(x)

        x = torch.flatten(self.avgpool(x), 1)
        out.append(x)

        return out

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.fill_(0)

class ResNet14(nn.Module):
    def __init__(self, in_size = 256):
        super().__init__()
        self.feature_extractor = ResNet()

        feat_size = 512
        side_ic = [64, 64, 128, 256, 512]
        side_oc = [1, 4, 16, 64, 256]
        side_nx = [4, 4, 8, 16, 32]

        # leves from 0, 1, 2, 3, 4 
        self.levels = [1, 2, 3, 4]

        self.side_modules = nn.ModuleList()
        self.attens = nn.ModuleList()
        input_size = 0
        for l in self.levels:
            size_fc = side_oc[l] * (in_size//side_nx[l])**2
            input_size += size_fc

            self.side_modules.append(
                nn.Sequential(
                    nn.Conv2d(side_ic[l], side_oc[l], 3, 1, 1),
                    nn.ReLU(),
                )
            )
            self.attens.append(Attention(side_ic[l], 64))

        # the output features of pooling 
        side_fc = nn.Sequential(
            nn.Linear(512, feat_size),
            nn.ReLU(),
        )
        input_size += feat_size
        self.side_modules.append(side_fc)
       
        self.gaze_fc = nn.Sequential(
            nn.Linear(input_size, feat_size), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(feat_size, 2))

        # weights initialization
        # self.side_modules.apply(weight_init)
        # self.attens.apply(weight_init)
        # self.gaze_fc.apply(weight_init)


    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.feature_extractor(x)

        x_sides = []
        for i, l in enumerate(self.levels):
            atten = self.attens[i](out[l])
            side = self.side_modules[i](out[l]) * atten # out
            side = torch.flatten(side, 1)
            x_sides.append(side)
        
        side = self.side_modules[-1](out[-1])
        x_sides.append(side)
        x = torch.cat(x_sides, dim=1)

        gaze_angles = self.gaze_fc(x)
        # gaze_angles1 = self.gaze_fc1(x_side1)
        # gaze_angles2 = self.gaze_fc2(x_side2)
        # gaze_angles3 = self.gaze_fc3(x_side3)
        # gaze_angles4 = self.gaze_fc4(x_side4)
        # gaze_angles5 = self.gaze_fc5(x_side5)

        return gaze_angles#, gaze_angles1, gaze_angles2, gaze_angles3, gaze_angles4, gaze_angles5


if __name__ == '__main__':

    x = torch.rand(64, 3, 256, 256)

    net = ResNet14()


    out = net(x)

    print(out.shape)
