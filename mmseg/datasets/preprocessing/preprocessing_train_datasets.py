import os
import torch.distributed as dist

from .decathlon import preprocess_decathlon_train
from mmseg.utils import is_master


def prepare_train_datasets(cfg, logger, ratio=0.2):
    logger.info('Preparing dataset {}'.format(cfg.dataset_type))
    if cfg.use_tmp_dir:
        if 'TMPDIR' not in os.environ:
            new_dataset_root = cfg.tmp_dir
        else:
            new_dataset_root = os.getenv('TMPDIR')
        if cfg.distributed:
            if is_master():
                preprocess_data(cfg, new_dataset_root, logger, ratio=ratio)
                dist.barrier()
            else:
                dist.barrier()
        else:
            preprocess_data(cfg, new_dataset_root, logger, ratio=ratio)
    else:
        if not 'WORK_ROOT' in os.environ:
            work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        else:
            work_dir = os.getenv('WORK_ROOT')
        new_dataset_root = os.path.join(work_dir, cfg.data_root)
    cfg.data_root = new_dataset_root
    cfg.data.train.data_root = new_dataset_root
    cfg.data.val.data_root = new_dataset_root
    return cfg


def preprocess_data(cfg, new_dataset_root, logger, ratio=0.2):
    if cfg.dataset_base == 'Decathlon':
        preprocess_decathlon_train(cfg, new_dataset_root, logger, ratio=ratio)
