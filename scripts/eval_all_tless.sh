#!/bin/bash

checkpoint=results/pkpnet_tless_primesense+pbr/checkpoint-88.pth.tar

model_dir=`dirname $checkpoint`
# LOG the outputs
log="${checkpoint%.pth.tar}_eval.log"
echo ================= LOG FILE ===================== > $log
echo Logging to $log

# NOTE you will have to manually kill this process if you kill the script before is kills $log_PID
tail -f $log &
log_PID=$!
echo LOG PID = $log_PID

base_args="--dataset tless -c $checkpoint $@"

# Single view
python3 -u evaluate.py $base_args --nviews 1 --no_viz >> $log 2>&1

# SLAM with more visualizations
#python3 -u evaluate.py $base_args --nviews -1 --viz_cov --do_viz_extra --show_viz >> $log 2>&1

# SLAM
python3 -u evaluate.py  $base_args --nviews -1 --gt_cam_pose >> $log 2>&1
python3 -u evaluate.py  $base_args --nviews -1 --gt_cam_pose --no_prior_det >> $log 2>&1
python3 -u evaluate.py  $base_args --nviews -1 --gt_cam_pose --no_network_cov >> $log 2>&1
#python3 -u evaluate.py $base_args --nviews -1 >> $log 2>&1

# Debug with GT keypoints
#python3 -u evaluate.py $base_args --nviews 1 --debug_gt_kp >> $log 2>&1
#python3 -u evaluate.py $base_args --nviews -1 --debug_gt_kp >> $log 2>&1
#python3 -u evaluate.py $base_args --nviews -1 --gt_cam_pose --debug_gt_kp >> $log 2>&1

kill $log_PID

notify-send Bash "All eval completed!"

# Print the results
echo Results table > $model_dir/table.txt
echo  > $model_dir/table.txt
for dir in $model_dir/pkpnet*; do
    echo $dir
    cat $dir/summary.txt
    echo
    
    echo $dir >> $model_dir/table.txt
    cat $dir/summary.txt >> $model_dir/table.txt
    echo  >> $model_dir/table.txt
done
