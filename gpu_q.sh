var=0
while [ $var -eq 0 ]
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 500 ]
        then
            echo 'GPU'$count' is avaiable'
            sleep 10
            ./mmdetection/tools/dist_train.sh mmdetection/configs/icartoonface/fr50_lite_dcn_gn_icf_rpn05_ms49_1549_2x.py 8 --seed 0
            var=1
            break
        fi
        count=$(($count+1))
    done
    sleep 10
done