_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/lits17.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2, # Fixed
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        # CrossEntropyLoss
        # loss_decode=dict(
        #     type='CrossEntropyLoss', 
        #     use_sigmoid=False, 
        #     loss_weight=1.0)),

        # CrossEntropyLoss + DiceLoss
        loss_decode=dict(
            type='CrossEntropyDiceLoss',
            ce_cfg=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0),
            dice_weight=2.0
        )),

        # FocalTversky and DiceLoss
        # loss_decode=dict(
        #     type='FocalTverskyDiceLoss',
        #     alpha=0.7,
        #     beta=0.3,
        #     gamma=0.75,
        #     loss_weight=1.0
        # )),

        # FocalTversky, DiceLoss and Boundary Loss
        # loss_decode=dict(
        #     type='FocalTverskyDiceBoundaryLoss',
        #     alpha=0.7,
        #     beta=0.3,
        #     gamma=0.75,
        #     lambda_dice=2.0,
        #     lambda_boundary=1.0
        # )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00003, # 0.00006
    betas=(0.9, 0.999), 
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'pos_block': dict(decay_mult=0.),
                    'norm': dict(decay_mult=0.),
                    # 'head': dict(lr_mult=10.)}))
                    'head': dict(lr_mult=5.)}))

lr_config = dict(_delete_=True, 
                 policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, 
                 min_lr=0.0, 
                 by_epoch=False)


data = dict(samples_per_gpu=4, workers_per_gpu=2)
evaluation = dict(interval=3000, metric='mDice')