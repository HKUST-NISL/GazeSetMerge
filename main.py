import torch
from trainer import Trainer

import numpy as np

from utils import load_config, get_output_dir, save_config


if __name__ == '__main__':
    config = load_config()
    output_dir = get_output_dir(config)

    if not config.is_train:
        save_config(config, output_dir)

    # instantiate trainer
    trainer = Trainer(config, output_dir)

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()
