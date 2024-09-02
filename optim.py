from typing import Any, Dict, List

import torch
import yacs.config

def create_optimizer(config: yacs.config.CfgNode,
                     model: torch.nn.Module) -> Any:
    # params = get_param_list(config, model)
    # params = [p for p in model.parameters() if p.requires_grad]

    base_params = []
    calib_params = []
    for name, param in model.named_parameters():
        # print(name)
        if 'convertor' not in name:
            base_params.append(param)
        else:
            calib_params.append(param)
    # print(len(base_params), len(calib_params))
    base_lr = config.base_lr
    params = [{'params': base_params, 'lr': base_lr},
              {'params': calib_params, 'lr': base_lr}]#, 'weight_decay': 0.0}]

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, 
            lr=config.base_lr,  
            momentum=config.momentum, 
            weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, 
            lr=config.base_lr,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, 
            lr=config.base_lr, 
            weight_decay=config.weight_decay)
    else:
        raise ValueError()
    return optimizer