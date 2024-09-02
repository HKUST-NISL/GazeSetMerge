import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class MultiSpacial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_layers, layer_idx=[0, 1, 2], rs_target=8):

        xx_layers = []
        for id in layer_idx:
            x = x_layers[id]
            b, c, h, w = x.shape
            if rs_target != h:
                x = F.avg_pool2d(x, h//rs_target)
            
            xx_layers.append(x)
        
        output = torch.cat(xx_layers, dim=1)

        return output
