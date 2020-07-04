#!/usr/bin/env bash
python main.py \
--method pt_and_ft \
--pt_loss MoCo \
--train_list ../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt \
--val_list ../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--arch i3d \
--mode rgb \
--lr 0.003 \
--optim sgd \
--spatial_size 224 \
--lr_steps 15 20 25 \
--epochs 30 \
--batch_size 6 \
--data_length 16 \
--dropout 0.5 --gpus 3 \
--stride 4 \
--logs_path ../experiments/logs/hmdb51_i3d_pt_and_ft