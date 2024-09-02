import argparse
import yacs.config
from yacs.config import CfgNode

config = CfgNode()

config.model = 'resnet14'
config.output_dir = './experiments/gaze/'


# data
config.data_type = 'mpii' #'xgaze'
config.gaze_data = './data/MPIIFaceGaze_norm.h5'
config.label_factor = 1

config.train_tag = ''
config.base_dir = ''
config.test_tag = ''
config.is_train = True
config.train_test = False
config.input_size = [224, 224]
config.mirror = True
config.rotate = 4.
config.scale = 0.08
config.offset = 12
config.mean = [0.406, 0.456, 0.485]
config.std = [0.225, 0.224, 0.229]

# train
config.gpu_id = 0
config.batch_size = 64
config.iter_size = 1
config.epochs = 15
config.base_lr = 0.001
config.test_step = 5
config.save_step = 5
config.loss = 'L2'
config.log_step = 200
config.finetune = ''
config.stop_thresh = 0.
config.with_la = False
config.companion = False
config.warmup = 0
config.validation = False

# schedule
config.scheduler_type = 'step'
config.lr_stepsize = 10
config.lr_milestones = [3, 10, 13]
config.gamma = 0.1

# optimizer
config.optimizer = 'adam'
config.weight_decay = 1e-4
config.momentum = 0.9

# test
config.test_ids = []
config.test_batch_size = 32
config.test_model = ''
config.test_data = ''
config.test_label = False
config.test_output = ''
config.test_eth = False


def get_default_config():
    return config.clone()

