import torch
from torch import nn

def create_loss(args):

    if args.loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    return criterion
