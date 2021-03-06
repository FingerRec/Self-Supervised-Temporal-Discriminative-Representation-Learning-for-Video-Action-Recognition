#!/usr/bin/env bash
python feature_extractor.py \
--eval_indict feature_extract \
--method ft \
--train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--arch i3d \
--mode rgb \
--lr 0.001 \
--lr_steps 10 20 25 30 35 40 \
--epochs 45 \
--batch_size 1 \
--data_length 64 \
--spatial_size 224 \
--workers 8 \
--dropout 0.5 \
--gpus 0 \
--logs_path ../experiments/logs/hmdb51_i3d_ft \
--print-freq 100  \
--weights ../experiments/MoCo/ucf101/models/03-22-1128_aug_CJ/ckpt_epoch_132.pth
#--weights ../experiments/logs/hmdb51_i3d_ft/ft_03-23-2206/fine_tune_rgb_model_latest.pth.tar

#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-18-1837/fine_tune_rgb_model_best.pth.tar
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-19-1046/fine_tune_rgb_model_best.pth.tar
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-19-1058/fine_tune_rgb_model_best.pth.tar
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-19-1058/net_mixup_rgb_model_best.pth.tar
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-19-1046/flip_cls_rgb_model_best.pth.tar
#--weights ../experiments/pretrained_model/model_rgb.pth
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-15-1229/mutual_loss_rgb_model_latest.pth.tar