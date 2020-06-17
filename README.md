0.以下所有操作的工作目录都默认为icartoonface_v2
将icartoonface数据放置在../data/icartoonface/下。zip包放在该目录下直接解压就行。
将widerface的train part数据放置在../data/WiderFace/下。同理将zip包放置在该目录下直接解压就行。

1.参考https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md 完成mmdet的编译安装
git clone mmdetection这一步替换为直接进入我的icartoonface_v2/mmdetection
补充安装一个jupyterlab。conda install -c conda-forge jupyterlab

2.生成数据的pkl描述文件
python prepare_det.py

训练机器配置 8卡 单卡16G或者以上显存的gpu机器。
3.训练单尺度。作为后续finetune的一个pretrained model weight
执行
./mmdetection/tools/dist_train.sh mmdetection/configs/icartoonface/fr50_lite_dcn_gn_scratch_icf_wf.py 8 --seed 0
参考：8v100 16G机器训练大概需要20小时

4.finetune模型。加入多尺度，mixup等
./mmdetection/tools/dist_train.sh mmdetection/configs/icartoonface/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x_40e.py 8
 --seed 0
参考：训练大概耗时17小时

5.在测试图上infereence。下面的epoch_x替换为上面finetune的时候验证集上表现最好的epoch。(我的实验中35ep最优)
./mmdetection/tools/dist_test.sh mmdetection/configs/icartoonface/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x_40e.py work_dirs/fr50_lite_dcn_gn_icf_ms49_1549_mixup_smooth_2x_40e/epoch_x.pth 8 --out  test.pkl

6.在jupyter lab中打开submit_size_limit_filter.ipynb。直接运行得到格式为submit{DATA}.csv格式的提交文件。

7.关于fps测试。queue_test_time.ipynb文件是按照比赛要求构建的测试模板。替换为对应的config文件和pth文件路径，执行之后查看wall time即2000张1920*1080图片的测试耗时。2000张图的测试时间应该为1分20-30秒之间。

8.关于模型大小。上传的模型大小为226M，因为其中保存了训练的中间状态，方便做实验的时候从中间结果恢复训练。给到官方复现的config文件里面已经改写为save_optimizer=False。实际模型为46M。
