#!/usr/bin/env bash
python main.py \
--method pt_and_ft \
--pt_loss triplet \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch i3d \
--mode rgb \
--lr 0.01 \
--lr_steps 4 8 11 13 \
--epochs 15 \
--batch_size 3 \
--spatial_size 112 \
--dropout 0.5 --gpus 0 \
--logs_path ../experiments/logs/hmdb51_i3d_pt_and_ft  --workers 3