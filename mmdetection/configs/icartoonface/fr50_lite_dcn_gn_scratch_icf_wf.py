_base_ = './fr50_lite_gn_scratch_icf_wf.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))