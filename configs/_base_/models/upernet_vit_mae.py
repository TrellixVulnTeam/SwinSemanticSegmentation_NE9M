# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ViTMAE',
        image_size=224,
        dim=1024,
        depth=12,
        heads=12,
        patch_size=16,
        channels=1,
        num_classes=1000,
        mlp_dim=2048,
        out_indices=[3, 5, 7, 11]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
