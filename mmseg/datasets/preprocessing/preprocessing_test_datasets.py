import os
import torch.distributed as dist

from .decathlon import preprocess_decathlon_test
from mmseg.utils import is_master

def prepare_test_datasets(cfg):
    print('Preparing dataset {}'.format(cfg.dataset_type))
    if 'TMPDIR' not in os.environ:
        new_dataset_root = cfg.tmp_dir
    else:
        new_dataset_root = os.getenv('TMPDIR')
    if is_master():
        preprocess_data(cfg, new_dataset_root)
        dist.barrier()
    else:
        dist.barrier()
    cfg.data_root = new_dataset_root
    cfg.data.test.data_root = new_dataset_root
    return cfg


def preprocess_data(cfg, new_dataset_root):
    if cfg.dataset_base == 'Decathlon':
        preprocess_decathlon_test(cfg, new_dataset_root)
