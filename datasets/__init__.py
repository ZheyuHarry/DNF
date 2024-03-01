import importlib
from copy import deepcopy
from os import path as osp
from glob import glob
import mindspore
from mindspore import dataset

from utils.registry import DATASET_REGISTRY

__all__ = ['build_train_loader','build_valid_loader','build_test_loader']

# automatically scan and import dataset modules for registry
# scan all the files under the 'datasets' folder and collect files ending with '_dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(dataset_folder, '*_dataset.py'))]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]

def build_dataset(dataset_cfg, split: str):
    assert split.upper() in ['TRAIN', 'VALID', 'TEST']
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    process_cfg = dataset_cfg.pop('process')
    split_cfg = dataset_cfg.pop(split)
    print(dataset_cfg , '\n\n\n')
    print(process_cfg, '\n\n\n')
    print(split_cfg, '\n\n\n')
    print(DATASET_REGISTRY.get(dataset_type))
    dataset = DATASET_REGISTRY.get(dataset_type)(
        **dataset_cfg,
        **process_cfg,
        **split_cfg,
        split=split
    )
    print(dataset)
    return dataset

def build_train_loader(dataset_cfg):
    train_dataset = build_dataset(dataset_cfg, 'train')
    print(f"train_dataset    {train_dataset}")
    train_dataloader = dataset.GeneratorDataset(source=train_dataset, shuffle=True,
                                                num_parallel_workers=dataset_cfg['num_workers'],
                                                )
    train_dataloader = train_dataloader.batch(dataset_cfg['train']['batch_size'], drop_remainder=False)
    train_dataloader = train_dataloader.create_tuple_iterator(num_epochs=dataset_cfg['persistent_workers'])

    return train_dataloader

def build_valid_loader(dataset_cfg, num_workers=None):
    valid_dataset = build_dataset(dataset_cfg, 'valid')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    valid_dataloader = dataset.GeneratorDataset(source=valid_dataset, shuffle=False, num_parallel_workers=num_workers)
    valid_dataloader = valid_dataloader.batch(dataset_cfg['valid']['batch_size'], drop_remainder=False)
    valid_dataloader = valid_dataloader.create_tuple_iterator(num_epochs=dataset_cfg['persistent_workers'])

    return valid_dataloader


def build_test_loader(dataset_cfg, num_workers=None):
    test_dataset = build_dataset(dataset_cfg, 'test')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']

    test_dataloader = dataset.GeneratorDataset(source=test_dataset, shuffle=False, num_parallel_workers=num_workers)
    test_dataloader = test_dataloader.batch(dataset_cfg['test']['batch_size'], drop_remainder=False)
    test_dataloader = test_dataloader.create_tuple_iterator(num_epochs=dataset_cfg['persistent_workers'])

    return test_dataloader