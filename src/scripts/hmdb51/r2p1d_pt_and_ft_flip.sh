#!/usr/bin/env bash
python main.py \
--method pt_and_ft \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch r2p1d \
--mode rgb \
--lr 0.001 \
--lr_steps 4 8 11 13 \
--epochs 15 \
--batch_size 1 \
--dropout 0.5 --gpus 0 \
--stride 1 \
--logs_path ../experiments/logs/hmdb51_i3d_pt_and_ft