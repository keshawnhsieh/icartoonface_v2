#!/usr/bin/env bash


config_file='mmdetection/configs/icartoonface/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x_40e.py'
checkpoint_file='work_dirs/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x_40e/epoch_35.pth'

last_thr=5
for ((thr=40;thr<=60;thr+=1));
do
    echo ${thr}
    sed -i "s/iou_thr=0.${last_thr}/iou_thr=0.${thr}/g" ${config_file};

    last_thr=${thr}

    ./mmdetection/tools/dist_test.sh ${config_file} ${checkpoint_file} 8 --eval mAP
done


# PLZ run with
# bash grid_search_nms.sh |tee grid_search_nms.log