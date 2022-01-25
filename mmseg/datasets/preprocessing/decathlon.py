import nibabel as nib
import numpy as np
from PIL import Image
import os
from .utils import multiprocess_data_and_labels, split_val_from_train, clip_ct_window, multiprocess_data, clip_ct_window_cube_root


def preprocess_decathlon_train(cfg, new_dataset_root, logger):
    if not 'WORK_ROOT' in os.environ:
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    else:
        work_dir = os.getenv('WORK_ROOT')
    vol_root = os.path.join(work_dir, cfg.data_root, 'imagesTr')
    label_root = os.path.join(work_dir, cfg.data_root, 'labelsTr')

    volumes = os.listdir(vol_root)
    volumes = [os.path.join(vol_root, f) for f in volumes]
    labels = os.listdir(label_root)
    labels = [os.path.join(label_root, f) for f in labels]
    assert len(volumes) == len(labels)

    train_data, train_labels, val_data, val_labels = split_val_from_train(volumes, labels, cfg.seed)
    logger.info('Split dataset of size {} into train: {} and val: {}'.format(len(volumes), len(train_data), len(val_data)))

    logger.info('Creating new directories at new data root{}'.format(new_dataset_root))

    # Prepare output dirs
    new_img_train_dir = os.path.join(new_dataset_root, 'img_dir', 'train')
    new_label_train_dir = os.path.join(new_dataset_root, 'ann_dir', 'train')
    new_img_val_dir = os.path.join(new_dataset_root, 'img_dir', 'val')
    new_label_val_dir = os.path.join(new_dataset_root, 'ann_dir', 'val')
    os.makedirs(new_img_train_dir, exist_ok=True)
    os.makedirs(new_label_train_dir, exist_ok=True)
    os.makedirs(new_img_val_dir, exist_ok=True)
    os.makedirs(new_label_val_dir, exist_ok=True)

    logger.info('Preparing training data')
    eval = False
    multiprocess_data_and_labels(process_volume_and_label, train_data, train_labels, new_img_train_dir, new_label_train_dir,
                      cfg.ct_window[0], cfg.ct_window[1], logger, eval)

    logger.info('Preparing validation data')
    eval = True
    multiprocess_data_and_labels(process_volume_and_label, val_data, val_labels, new_img_val_dir, new_label_val_dir,
                      cfg.ct_window[0], cfg.ct_window[1], logger, eval)
    logger.info('Preprocessing of Decathlon data complete')


def preprocess_decathlon_test(cfg, new_dataset_root):
    if not 'WORK_ROOT' in os.environ:
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    else:
        work_dir = os.getenv('WORK_ROOT')
    vol_root = os.path.join(work_dir, cfg.data_root, 'imagesTs')

    volumes = os.listdir(vol_root)
    volumes = [os.path.join(vol_root, f) for f in volumes]

    print('Found test dataset of size {}'.format(len(volumes)))

    print('Creating new directories at new data root{}'.format(new_dataset_root))

    # Prepare output dirs
    new_img_test_dir = os.path.join(new_dataset_root, 'img_dir', 'test')
    os.makedirs(new_img_test_dir, exist_ok=True)

    print('Preparing test data')
    multiprocess_data(process_volume, volumes, new_img_test_dir, cfg.ct_window[0], cfg.ct_window[1])

    print('Preprocessing of Decathlon data complete')


def process_volume_and_label(vol_file, label_file, img_dir, label_dir, ct_min, ct_max, logger, eval):
    try:
        vol_data = nib.load(vol_file)
        label_data = nib.load(label_file)
        np_vol = vol_data.get_fdata()
        np_label = label_data.get_fdata()
        assert np_vol.shape == np_label.shape
        np_vol = clip_ct_window_cube_root(np_vol, ct_min, ct_max)
        np_vol = np.transpose(np_vol, (2, 0, 1)).astype(np.uint8)
        np_label = np.transpose(np_label, (2, 0, 1)).astype(np.uint8)
        for j, (v_slice, l_slice) in enumerate(zip(np_vol, np_label)):
            # Only include slices with labels other than background
            if eval:
                org_name = os.path.split(vol_file)[1][:-7]
                slice_name = org_name + '_' + str(j) + '.png'
                img_out_path = os.path.join(img_dir, slice_name)
                label_out_path = os.path.join(label_dir, slice_name)
                pil_img = Image.fromarray(v_slice)
                pil_ann = Image.fromarray(l_slice)
                # pil_ann = pil_ann.convert('P', palette=Image.ADAPTIVE, colors=3)
                pil_img.save(img_out_path)
                pil_ann.save(label_out_path)
            else:
                if 2 in l_slice:
                    org_name = os.path.split(vol_file)[1][:-7]
                    slice_name = org_name + '_' + str(j) + '.png'
                    img_out_path = os.path.join(img_dir, slice_name)
                    label_out_path = os.path.join(label_dir, slice_name)
                    pil_img = Image.fromarray(v_slice)
                    pil_ann = Image.fromarray(l_slice)
                    # pil_ann = pil_ann.convert('P', palette=Image.ADAPTIVE, colors=3)
                    pil_img.save(img_out_path)
                    pil_ann.save(label_out_path)
        return 0
    except Exception as e:
        logger.error('Failed to processes volume {} and label {} with error {}'.format(v, l, e))
        return 1


def process_volume(vol_file, img_dir, ct_min, ct_max):
    try:
        vol_data = nib.load(vol_file)
        np_vol = vol_data.get_fdata()
        np_vol = clip_ct_window_cube_root(np_vol, ct_min, ct_max)
        np_vol = np.transpose(np_vol, (2, 0, 1)).astype(np.uint8)
        for j, v_slice in enumerate(np_vol):
            org_name = os.path.split(vol_file)[1][:-7]
            slice_name = org_name + '_' + str(j) + '.png'
            img_out_path = os.path.join(img_dir, slice_name)
            pil_img = Image.fromarray(v_slice)
            # pil_ann = pil_ann.convert('P', palette=Image.ADAPTIVE, colors=3)
            pil_img.save(img_out_path)
        return 0
    except Exception as e:
        print('Failed to processes volume {} with error {}'.format(v, e))
        return 1
