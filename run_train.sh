#!/usr/bin/env bash

#data-set
model_name="detection"
modality="Flow"
DATE=`date '+%Y-%m-%d-%H-%M-%S'`

mkdir -p ./nohups/$model_name/$modality
mkdir -p ./models/$model_name/$modality/$DATE

nohup python trainval.py $model_name $modality \
        --data_path /home/ly/workspace/trecvid/data/person_vehicle_interaction \
        --pretrained_weights kinetics_flow.pth \
        --train_list train_label_file.txt \
        --val_list val_label_file.txt \
        --arch S3DG --gpus 0 1 \
        --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
        -b 48 -d 8 --dropout 0.5 \
        --snapshot_pref ./models/$model_name/$modality/$DATE/epoch \
> nohups/$model_name/$modality/${DATE}-log.out 2>&1 &
