import pathlib
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

class OnePersonDataset(Dataset):
    def __init__(self, person_id_str, dataset_path, transform=None, reduce_flip=False, index_list=None):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.reduce_flip = reduce_flip
        self.index_list = index_list

    def __getitem__(self, index):

        if self.index_list is not None:
            index = self.index_list[index]

        with h5py.File(self.dataset_path, 'r') as f:
            face = f.get(f'{self.person_id_str}/face/{index:04}')[()]
            lmk = f.get(f'{self.person_id_str}/lmk/{index:04}')[()]
            left_eye_box = f.get(f'{self.person_id_str}/left_eye_box/{index:04}')[()]
            right_eye_box = f.get(f'{self.person_id_str}/right_eye_box/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]

            subid = int(self.person_id_str[1])

        x = {'image': face, 'gaze': gaze, 'sub_id': subid, 'lmk': lmk, 'left_eye_box':left_eye_box, \
            'right_eye_box':right_eye_box, 'pose':pose}

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        if self.index_list is None:
            return 3000
        else:
            return len(self.index_list)

def mpii_gaze_dataset(data_dir, test_id=0, transform=None, is_train=True):

    dataset_dir = pathlib.Path(data_dir)

    assert dataset_dir.exists()
    assert test_id in range(-1, 15)
    assert test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]


    if is_train:
    
        test_person_id = person_ids[test_id]
        train_dataset = torch.utils.data.ConcatDataset([
            OnePersonDataset(person_id, dataset_dir, transform)
            for person_id in person_ids if person_id != test_person_id
        ])
        assert len(train_dataset) == 42000
        return train_dataset


    else:
        test_person_id = person_ids[test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform)
        assert len(test_dataset) == 3000
        return test_dataset


def draw_eye_box(img, box):
    box = box.copy().astype(np.int32)
    img = img.copy()
    
    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    return img


if __name__ == '__main__':


    for id in range(15):
        test_id = id
        data = mpii_gaze_dataset(data_dir='./data/processed/MPIIFaceGaze_norm.h5', test_id=test_id, is_train=False)

        print(len(data))
        faces = []
        gazes = []


        for i, x in enumerate(data):
            face = x['image']
            gaze = x['gaze']/np.pi*180
            pose = x['pose']/np.pi*180
            left_eye_box = x['left_eye_box']
            right_eye_box = x['right_eye_box']
            # print(i, ang, pose)
            faces.append(face)
            gazes.append(gaze)
            face = draw_eye_box(face, left_eye_box)
            face = draw_eye_box(face, right_eye_box)
            cv2.imshow('face', face)
            cv2.waitKey(0)

        a = np.array(gazes).reshape(-1, 2)





