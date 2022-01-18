_base_ = [
    '../_base_/models/upernet_vit_mae_v2.py', '../_base_/datasets/decathlon_liver.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=3,
        channels=768
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=3
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
#optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                 'relative_position_bias_table': dict(decay_mult=0.),
#                                                 'norm': dict(decay_mult=0.)}))

optimizer = dict(_delete_=True, type='AdamW', lr=7e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8)
