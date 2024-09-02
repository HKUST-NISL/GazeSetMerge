import os
import pathlib

import numpy as np
import torch
import tqdm

from utils import load_config, AverageMeter
from models import create_model
from datasets import create_gaze_dataloader
from utils import compute_angle_error

def test_gaze(model, test_loader, config):

    gpu = config.gpu_id
    model.eval()

    predictions = []
    gts = []
    with torch.no_grad():
        for data in test_loader:
            images = data['face'].cuda(gpu)
            label_angles = data['gaze'].cuda(gpu)

            gaze_angles = model(images)
            if type(gaze_angles) == tuple:
                gaze_angles = gaze_angles[0]

            predictions.append(gaze_angles.cpu() * (1. / config.label_factor))
            gts.append(label_angles.cpu())


    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())

    py_errors = torch.mean(torch.abs(predictions-gts), dim=0) / np.pi * 180
    return angle_error, py_errors

def test_xgaze(model, test_loader, config):

    gpu = config.gpu_id
    model.eval()

    avg_errors = {}
    with torch.no_grad():
        for data in test_loader:
            sub_ids = data['sub_id']
            images = data['face'].cuda(gpu)
            label_angles = data['gaze'].cuda(gpu)

            gaze_angles = model(images)
            if type(gaze_angles) == tuple:
                gaze_angles = gaze_angles[0]

            predictions = gaze_angles.cpu() * (1. / config.label_factor)
            gts = label_angles.cpu()
            errors = compute_angle_error(predictions, gts)

            num = len(sub_ids)
            for i in range(num):
                sub_id = sub_ids[i]
                if sub_id not in avg_errors.keys():
                    avg_errors[sub_id] = AverageMeter()
                avg_errors[sub_id].update(errors[i].item(), 1)
    
    mean_error = 0
    for key in avg_errors.keys():
        avg_errors[key] = avg_errors[key].avg
        mean_error += avg_errors[key]
    
    mean_error = mean_error / len(avg_errors)

    return mean_error, avg_errors

if __name__ == '__main__':
    config = load_config()
    # print(config)

    # load checkpoint
    checkpoint = torch.load(config.test_model, map_location='cpu')
    # for var_name in checkpoint:
    #     print(var_name, checkpoint[var_name].shape)

    model = create_model(config)
    # state_dict = model.state_dict()
    # for var_name in state_dict:
    #     print(var_name, state_dict[var_name].shape)

    model.load_state_dict(checkpoint)
    print('Load eval model from %s' % config.test_model)
    gaze_test_data = create_gaze_dataloader(config, False)

    # test
    error, py_errors = test_gaze(model, gaze_test_data, config)
    print('Test Gaze Test id: %02d Error: %.6f %.6f %.6f' % (config.test_id, error, py_errors[0], py_errors[1]))

