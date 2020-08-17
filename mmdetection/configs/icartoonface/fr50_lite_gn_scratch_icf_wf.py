_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/icartoonface_plus.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        base_channels=32,
        frozen_stages=-1, zero_init_residual=False, norm_cfg=norm_cfg),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ))
# optimizer
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(warmup_ratio=0.1, step=[65, 71])
total_epochs = 73
# resume_from='work_dirs/fr50_lite_gn_scratch_icf_wf/latest.pth'
