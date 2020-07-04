#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train_temporal_dis.py \
--batch_size 16 --num_workers 8 --nce_k 3569 --softmax --moco \
--print_freq 100 --dataset 'hmdb51' \
--train_list '../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt' \
--val_list '../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt'