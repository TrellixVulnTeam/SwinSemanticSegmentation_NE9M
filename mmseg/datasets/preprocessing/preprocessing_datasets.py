import os
from mmcv.runner import master_only

from decathlon import preprocess_decathlon


@master_only
def prepare_datasets(cfg, logger):
    logger.info('Preparing dataset {}'.format(cfg.dataset_type))
    if 'TMPDIR' not in os.environ:
        new_dataset_root = cfg.tmp_dir
    else:
        new_dataset_root = os.getenv('TMPDIR')
    if cfg.dataset_base == 'Decathlon':
        preprocess_decathlon(cfg, new_dataset_root, logger)
    cfg.data_root = new_dataset_root

