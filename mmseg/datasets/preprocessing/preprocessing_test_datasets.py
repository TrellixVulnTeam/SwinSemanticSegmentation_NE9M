import os
import torch.distributed as dist

from .decathlon import preprocess_decathlon
from mmseg.utils import is_master

def prepare_test_datasets(cfg, logger):
    logger.info('Preparing dataset {}'.format(cfg.dataset_type))
    if 'TMPDIR' not in os.environ:
        new_dataset_root = cfg.tmp_dir
    else:
        new_dataset_root = os.getenv('TMPDIR')
    if is_master():
        preprocess_data(cfg, new_dataset_root, logger)
        dist.barrier()
    else:
        dist.barrier()
    cfg.data_root = new_dataset_root
    cfg.data.test.data_root = new_dataset_root
    return cfg


def preprocess_data(cfg, new_dataset_root, logger):
    if cfg.dataset_base == 'Decathlon':
        preprocess_decathlon_test(cfg, new_dataset_root, logger)
