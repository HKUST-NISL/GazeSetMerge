from typing import Any

import torch
import yacs.config
from warmup_scheduler import GradualWarmupScheduler

def create_scheduler(config: yacs.config.CfgNode, optimizer: Any) -> Any:
    if config.scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_milestones,
            gamma=config.gamma)
    elif config.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_stepsize,
            gamma=config.gamma)
    elif config.scheduler_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=config.gamma)
    else:
        raise ValueError()


    if config.warmup > 0:
        scheduler = GradualWarmupScheduler( 
                    optimizer, 
                    multiplier=1, 
                    total_epoch=config.warmup, 
                    after_scheduler=scheduler
                    )
    return scheduler