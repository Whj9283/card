norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UnetBackbone',
        in_channels=3,
        channel_list=[64, 128, 256, 512]),
    decode_head=dict(
        type='UnetHead',
        num_classes=4,
        channels=64,
        threshold=0.2,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='L1Loss', reduction='none', loss_weight=2.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=2.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)]))

train_cfg = dict()
test_cfg = dict(mode='whole')
dataset_type = 'MyDataset'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(600, 600)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize',  mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = "./datasets/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
optimizer = dict(type='Adam', lr=1e-6, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=False)
runner = dict(
    type='IterBasedRunner',
    max_iters=5000)
checkpoint_config = dict(
    by_epoch=False,
    save_optimizer=False,
    interval=5000)
evaluation = dict(
    interval=500,
    metric=['mIoU', 'mFscore', 'mDice'])