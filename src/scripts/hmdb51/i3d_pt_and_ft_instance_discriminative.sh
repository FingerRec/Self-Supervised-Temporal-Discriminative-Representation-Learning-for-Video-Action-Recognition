#!/usr/bin/env bash
python main.py \
--method pt_and_ft \
--pt_loss instance_discriminative \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch i3d \
--mode rgb \
--lr 0.01 \
--optim sgd \
--spatial_size 224 \
--lr_steps 10 20 30 40 \
--epochs 45 \
--batch_size 12 \
--data_length 16 \
--dropout 0.5 --gpus 2 \
--stride 4 \
--logs_path ../experiments/logs/hmdb51_i3d_pt_and_ft