import os
import cv2
import numpy as np
import h5py
from scipy.io import loadmat
from pathlib import Path

import argparse
import pathlib
import tqdm

face_w = 448
face_h = 448
eye_w = 120
eye_h = 80
sub_num = 15
data_dir = '/data/datasets/MPIIFaceGaze_normalizad/'

def draw_points(img, pts):
    img = img.copy()
    pts2d = pts.copy().astype(np.int32)
    pts2d = pts2d.reshape(-1, 2)

    for i in range(len(pts2d)):
        img = cv2.circle(img, (pts2d[i, 0], pts2d[i, 1]), 2, (255,255,255), cv2.FILLED)

    return img

def get_rect(points, ratio=1.0, scale=1):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (max(x) - min(x))
    y_expand = 0.1 * (max(y) - min(y))

    x_max, x_min = max(x) + x_expand, min(x) - x_expand
    y_max, y_min = max(y) + y_expand, min(y) - y_expand

    # h:w=1:2
    if (y_max - y_min) * ratio < (x_max - x_min):
        h = (x_max - x_min) / ratio
        pad = (h - (y_max - y_min)) / 2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max - y_min)
        pad = (h * ratio - (x_max - x_min)) / 2
        x_max += pad
        x_min -= pad

    int(x_min), int(x_max), int(y_min), int(y_max)
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    bbox = np.array(bbox)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (448*scale, 448*scale))
    rect = np.concatenate([aSrc, bSrc])

    return rect


def add_mat_data_to_hdf5(person_id, dataset_dir: pathlib.Path,
                         output_path: pathlib.Path) -> None:

    mat_path = dataset_dir / f'{person_id}.mat'
    lmd_path = dataset_dir / 'landmarks' / f'{person_id}.mat'


    arrays = {}
    f = h5py.File(mat_path, 'r')
    for k, v in f.items():
        arrays[k] = v

    hdf_lmk = h5py.File(lmd_path, 'r')
    

    images = arrays['Data']['data']
    labels = arrays['Data']['label']
    gazes = labels[:, :2]
    poses = labels[:, 2:4]

    faces = []
    leye_boxes = []
    reye_boxes = []
    lmks = []
    for i in tqdm.tqdm(range(len(images))):
        face = np.array(images[i]).astype(np.uint8).transpose(1, 2, 0)
        label = np.array(labels[i])

        lmk = hdf_lmk['landmark'][i].copy()
        left_eye_box = get_rect(lmk[42:47], scale=1.)
        right_eye_box = get_rect(lmk[36:41], scale=1.)

        # print(face.shape)
        # # face = cv2.flip(face, 1)
        # face = draw_points(face, pts)
 
        faces.append(face)
        leye_boxes.append(left_eye_box)
        reye_boxes.append(right_eye_box)

        lmks.append(lmk)
        
    with h5py.File(output_path, 'a') as f_output:
        for index, (face, lmk, left_eye_box, right_eye_box,
                    gaze, pose) in tqdm.tqdm(enumerate(zip(faces, lmks, leye_boxes, reye_boxes, gazes, poses)),
                                       leave=False):
            f_output.create_dataset(f'{person_id}/face/{index:04}', data=face)
            f_output.create_dataset(f'{person_id}/lmk/{index:04}', data=lmk)
            f_output.create_dataset(f'{person_id}/left_eye_box/{index:04}', data=left_eye_box)
            f_output.create_dataset(f'{person_id}/right_eye_box/{index:04}', data=right_eye_box)
            f_output.create_dataset(f'{person_id}/gaze/{index:04}', data=gaze)
            f_output.create_dataset(f'{person_id}/pose/{index:04}', data=pose)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data_dir)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'MPIIFaceGaze_norm.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)
    for person_id in tqdm.tqdm(range(sub_num)):
        # person_id = 7
        person_id = f'p{person_id:02}'
        add_mat_data_to_hdf5(person_id, dataset_dir, output_path)


if __name__ == '__main__':
    main()
