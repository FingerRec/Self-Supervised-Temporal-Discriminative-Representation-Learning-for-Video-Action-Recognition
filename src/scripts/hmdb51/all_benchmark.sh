#!/usr/bin/env bash
for arch in r2p1d
do
    python main.py \
    --method ft \
    --train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
    --val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
    --dataset hmdb51 \
    --arch ${arch} \
    --mode rgb \
    --lr 0.01 \
    --lr_steps 10 20 25 30 35 40 \
    --epochs 45 \
    --batch_size 4 \
    --dropout 0.5 \
    --gpus 0 \
    --logs_path ../experiments/logs/hmdb51_${arch}_benchmark
done