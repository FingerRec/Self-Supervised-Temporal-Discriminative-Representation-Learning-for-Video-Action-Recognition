#!/usr/bin/env bash
for arch in i3d c3d r3d r2p1d
do
    python main.py \
    --method pt_and_ft \
    --train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
    --val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
    --dataset hmdb51 \
    --arch ${arch} \
    --mode rgb \
    --lr 0.01 \
    --lr_steps 4 8 11 13 \
    --epochs 15 \
    --batch-size 8 \
    --dropout 0.5 \
    --gpus 2 \
    --logs_path ../experiments/logs/hmdb51_${arch}_pt_and_ft
done