_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/icartoonface_plus_ms49_1549_mixup.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# fp16 settings
fp16 = dict(loss_scale=512.)

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, smoothing=0.001),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ))

test_cfg = dict(
    rcnn=dict(
        score_thr=0.00,
        nms=dict(type='nms', iou_thr=0.5,
                 # min_score=0.05
                 ),
        max_per_img=100))

data = dict(
    use_wider_face=False,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
load_from = 'work_dirs/fr50_lite_dcn_gn_scratch_icf_wf/epoch_66.pth'
checkpoint_config = dict(interval=1, save_optimizer=False)
total_epochs = 40
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 28])

# resume_from = 'work_dirs/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x/epoch_22.pth'