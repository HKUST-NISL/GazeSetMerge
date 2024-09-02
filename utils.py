import argparse
import pathlib
import random
import os
import sys

import numpy as np
import torch
import yacs.config

from config import get_default_config


def get_output_dir(config):
    tag = config.train_tag
    config.base_dir = os.path.join(config.output_dir, '%s/%s_%s' % (config.data_type, config.model, tag))
    if config.data_type == 'mpii' or config.data_type == 'diap':
        test_dir = '_'.join(['%02d' % (id) for id in config.test_ids])
        config.test_tag = test_dir
        output_dir = os.path.join(config.base_dir, '%s' % test_dir)
    elif config.data_type == 'eth':
        output_dir = config.base_dir

    if config.is_train:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('(Train) output dir created: %s' %  output_dir)
        else:
            print("(Train) output dir exists: %s" % (output_dir))
            sys.exit(0)

    return output_dir

def save_config(config: yacs.config.CfgNode, output_dir: pathlib.Path) -> None:
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(str(config))

def convert_to_unit_vector(angles):
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    eps = 0.000001
    norm = eps + norm
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def pitchyaw_to_vector_one(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((3))
    out[0] = np.multiply(cos[0], sin[1])
    out[1] = sin[0]
    out[2] = np.multiply(cos[0], cos[1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # pitch 
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # yaw
    return out

def vector_to_pitchyaw_one(vector):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    out = np.empty(2)
    vector = np.divide(vector, np.linalg.norm(vector)+0.000001)
    out[0] = np.arcsin(vector[1])  # pitch
    out[1] = np.arctan2(vector[0], vector[2])  # yaw
    return out

def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi


def compute_angle_error(predictions: torch.tensor,
                        labels: torch.tensor) -> torch.tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi

def compute_mae_errors(predictions, labels, radian=True):

    errors = torch.mean(torch.abs(predictions - labels), dim=0)

    if radian:
        errors = errors / 3.14 * 180
        
    mae = torch.mean(errors)

    return errors, mae

def load_config() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)

    # config.freeze()
    return config


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


if __name__ == '__main__':
    cfg = load_config()
    print(cfg)