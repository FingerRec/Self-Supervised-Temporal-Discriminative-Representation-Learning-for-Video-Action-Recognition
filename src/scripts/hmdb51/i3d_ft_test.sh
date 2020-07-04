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
--batch_size 4 \
--data_length 64 \
--spatial_size 224 \
--workers 4 \
--stride 1 \
--dropout 0.5 \
--gpus 3 \
--logs_path ../experiments/logs/hmdb51_i3d_ft \
--print-freq 100 \
--weights ../experiments/MoCo/ucf101/models/04-16-2328_aug_CJ/ckpt_epoch_96.pth
#--weights ../experiments/TemporalDis/kinetics/models/04-05-2337_aug_CJ/ckpt_epoch_33.pth
#--weights ../experiments/TemporalDis/ucf101/models/04-07-1545_aug_CJ/ckpt_epoch_72.pth
#--weights ../experiments/TemporalDis/ucf101/models/04-02-2147_aug_CJ/ckpt_epoch_99.pth
#--weights ../experiments/logs/ucf101_i3d_ft/ft_03-25-2142/fine_tune_rgb_model_latest.pth.tar
#--weights ../experiments/TemporalDis/ucf101/models/03-23-1128_aug_CJ/ckpt_epoch_57.pth
#--weights ../experiments/TemporalDis/kinetics/models/03-20-2211_aug_CJ/ckpt_epoch_6.pth
#--weights ../experiments/TemporalDis/ucf101/models/03-15-1209_aug_CJ/ckpt_epoch_69.pth
# --weights ../experiments/TemporalDis/ucf101/models/03-14-1536_aug_CJ/ckpt_epoch_30.pth
# --weights ../experiments/TemporalDis/hmdb51/models/03-13-1357_aug_CJ/ckpt_epoch_10.pth
#--weights ../experiments/TemporalDis/ucf101/models/03-13-1518_aug_CJ/ckpt_epoch_5.pth
# --weights ../experiments/TemporalDis/hmdb51/models/03-12-1706_aug_CJ/ckpt_epoch_30.pth %34.2
# --weights ../experiments/TemporalDis/hmdb51/models/03-12-1405_aug_CJ/ckpt_epoch_25.pth %32
#--weights ../experiments/pretrained_model/model_rgb.pth
#--weights ../experiments/logs/hmdb51_i3d_pt_and_ft/pt_and_ft_02-15-1229/mutual_loss_rgb_model_latest.pth.tar