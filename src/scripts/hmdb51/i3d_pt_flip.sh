#!/usr/bin/env bash
python main.py \
--method pt \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch i3d \
--mode rgb \
--lr 0.01 \
--lr_steps 10 20 25 30 35 40 \
--epochs 45 \
--batch-size 8 \
--dropout 0.5 \
--gpus 2 \
--logs_path ../experiments/logs/hmdb51_i3d_pt