#!/usr/bin/env bash
python main.py \
--method ft \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch i3d \
--mode rgb \
--lr 0.001 \
--lr_steps 10 20 25 30 35 40 \
--epochs 45 \
--batch_size 8 \
--spatial_size 224 \
--workers 8 \
--dropout 0.5 \
--gpus 1 \
--logs_path ../experiments/logs/hmdb51_i3d_ft \
--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_12-05-1310/flip_cls_rgb_model_best.pth.tar