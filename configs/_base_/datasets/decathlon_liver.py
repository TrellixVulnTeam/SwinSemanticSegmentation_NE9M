# dataset settings
dataset_base = 'Decathlon'
dataset_type = 'DecathlonLiverDataset'
data_root = 'data/Decathlon/3D/Task03_Liver'
decathlon_liver_mean = 0.1943
decathlon_livear_std = 0.2786
img_norm_cfg = dict(
    mean=[decathlon_liver_mean], std=[decathlon_livear_std], to_rgb=False)
crop_size = (224, 224)
pp_ct_window = True
ct_window = [-1000, 1000]
interpolate_voxel_spacing = False
voxel_spacing = 1.0
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.25, degree=36),
    dict(type='PhotoMetricDistortion'),
    dict(type='RescaleIntensity', scale_min=0, scale_max=1),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
'''
val_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=crop_size),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Resize', img_scale=crop_size),
    dict(type='RandomFlip', prob=0.0),
    dict(type='RescaleIntensity', scale_min=0, scale_max=1),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline)
    )
