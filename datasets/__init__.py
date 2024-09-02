from .all_data import AllData
from .one_data import OneData
from torch.utils.data import DataLoader

from datasets.data_transforms import create_transform

def create_gaze_dataloader(config, data_type= '', test_ids=[], is_train=True):

    transform = create_transform(config.input_size, is_train)

    if data_type == '':
        if is_train:
            train_dataset = AllData(data_dir=config.gaze_data, transform=transform)

            data_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4, 
                drop_last=True)

        else:
            None
    else:

        dataset = OneData(data_dir=config.gaze_data, data_type=data_type, test_ids=test_ids, transform=transform)

        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4, 
            drop_last=True)

    return data_loader

