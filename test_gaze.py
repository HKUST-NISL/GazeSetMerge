import os
import time
import pathlib

import numpy as np
import torch

from tqdm import tqdm

from utils import load_config, angular_error
from models import create_model
from datasets import create_gaze_dataloader
from datasets.one_data import create_one_loader
from utils import compute_angle_error, get_output_dir


def test_xgaze(model, test_loader, config):

    gpu = config.gpu_id
    model.eval()

    predictions = []
    gt_gazes = []
    avg_errors = {}
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for data in test_loader:
            pbar.update(1)

            if config.gpu_id >= 0:
                data = {k: v.cuda() for k, v in data.items()}
            pred_gaze = model(data)
            predictions.append(pred_gaze.cpu().detach().numpy())

            if config.test_data != 'eth':
                gaze = data['gaze'].cpu().detach().numpy()
                gt_gazes.append(gaze)

        pbar.close()
                
    predictions = np.concatenate(predictions, axis=0)
    
    if config.test_data != 'eth':
        gt_gazes = np.concatenate(gt_gazes, axis=0)
        err = angular_error(predictions, gt_gazes).mean()
        return predictions, err

    return predictions

if __name__ == '__main__':

    config = load_config()
    config.is_train = False
    if config.test_output == '':
        output_dir = get_output_dir(config)
    else:
        output_dir = config.test_output
    print(output_dir)

    if config.test_data == '':
        config.test_data = config.data_type

    test_data = config.test_data
    
    if config.test_model == '':
        test_model = os.path.join(output_dir, 'best_ckpt.pth.tar')
    else:
        test_model = config.test_model

    checkpoint = torch.load(test_model, map_location='cpu')
    model = create_model(config)
    if config.gpu_id >= 0:
        model = model.cuda(config.gpu_id)
    model.load_state_dict(checkpoint['model_state'], False)
    print('Load eval model from %s' % test_model)

    data_dir = config.gaze_data
    input_size = config.input_size
    batch_size = config.batch_size
    data_type = config.data_type
    test_ids = config.test_ids
    test_ids = []
    test_tag = '_'.join(['%02d' % (id) for id in config.test_ids])
    eth_test = False
    if test_data == 'eth':
        eth_test = True

    gaze_test_data = create_one_loader(data_dir, input_size, batch_size, test_data, test_ids, False, eth_test)
    print('Start test on %d samples' % len(gaze_test_data.dataset))

    # test
    predictions = test_xgaze(model, gaze_test_data, config)

    if test_data != 'eth':
        predictions, err = predictions
    print('Tested number of samples %d' % predictions.shape[0])

    out_dir = os.path.join(output_dir, test_data)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res_path = os.path.join(out_dir, 'within_eva_results.txt')
    np.savetxt(res_path, predictions, delimiter=',')
    print('Save results in %s' % (res_path))

    if test_data != 'eth':
        terr_dir = os.path.join(output_dir, '../%s' % (test_data))

        if not os.path.exists(terr_dir):
            os.makedirs(terr_dir)
        
        terr_path = os.path.join(terr_dir, 'errors.txt')
        
        with open(terr_path, 'a+') as f:
            err_line = '%s,%.6f\n' % (test_tag, err)
            f.write(err_line)
        print('Gaze error: %.4f saved in %s' %(err, terr_path))


