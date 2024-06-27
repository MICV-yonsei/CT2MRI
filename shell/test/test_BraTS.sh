#!/bin/bash
cd /root_dir/code

config_name="BBDM_base_BraTS.yaml"
HW="176"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_176_BBDM_axial_DDIM_BraTS_global_hist_context"

# test
test_epoch="34"
resume_model="/root_dir/code/results/BraTS_$HW/$exp_name/checkpoint/latest_model_$test_epoch.pth"
resume_optim="/root_dir/code/results/BraTS_$HW/$exp_name/checkpoint/latest_optim_sche_$test_epoch.pth"

sample_step=100
inference_type="normal" # normal, average, ISTA_average, ISTA_mid
ISTA_step_size=0.5
num_ISTA_step=1

python /root_dir/code/main.py \
    --exp_name $exp_name \
    --config /root_dir/code/results/BraTS_$HW/$exp_name/checkpoint/config_backup.yaml \
    --sample_to_eval \
    --gpu_ids $gpu_ids \
    --resume_model $resume_model \
    --resume_optim $resume_optim \
    --HW $HW \
    --plane $plane \
    --ddim_eta $ddim_eta \
    --sample_step $sample_step \
    --inference_type $inference_type \
    --ISTA_step_size $ISTA_step_size \
    --num_ISTA_step $num_ISTA_step


