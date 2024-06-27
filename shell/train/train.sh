#!/bin/bash
cd /root_dir/code

date="241213"

config_name="BBDM_base.yaml"
HW="160"
plane="axial"
gpu_ids="0"
batch=16
ddim_eta=0.0

prefix="MR_global_hist_context"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

mkdir /root_dir/code/results/ct2mr_${HW}/$exp_name

resume_model="/root_dir/code/results/ct2mr_$HW/$exp_name/checkpoint/last_model.pth"
resume_optim="/root_dir/code/results/ct2mr_$HW/$exp_name/checkpoint/last_optim_sche.pth"
python -u /root_dir/code/main.py \
    --train \
    --exp_name $exp_name \
    --config /root_dir/code/configs/$config_name \
    --dataset_type $dataset_type \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --sample_at_start \
    --save_top \
    --resume_model $resume_model \
    --resume_optim $resume_optim \
    --gpu_ids $gpu_ids

