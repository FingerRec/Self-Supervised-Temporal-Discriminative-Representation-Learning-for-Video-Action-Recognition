# Self-Supervised Temporal-Discriminative Representation Learning

The source code for our paper 

"Self-Supervised Temporal-Discriminative Representation
Learning for Video Action Recognition" [paper](https://arxiv.org/abs/2008.02129)

## Overview

> Our self-supervised VTDL signifcantly outperforms existing
self-supervised learning method in video action recognition, even achieve better result than fully-supervised methods on UCF101 and HMDB51
when a small-scale video dataset (with only thousands of videos) is
used for pre-training!

![sample_acc.png](https://i.loli.net/2020/07/04/WENzTnSv8f6cLyR.png)

### Requirements
- Python3
- pytorch1.1+
- PIL

## Structure
- datasets
    - list
        - hmdb51: the train/val lists of HMDB51
        - ucf101: the train/val lists of UCF101
        - kinetics-400: the train/val lists of kinetics-400
- experiments
    - logs: experiments record in detials
    - TemporalDis
        - hmdb51
        - ucf101
        - kinetics
    - gradientes: 
    - visualization
- src
    - data: load data
    - loss: the loss evluate in this paper
    - model: network architectures
    - scripts: train/eval scripts
    - TC: detail implementation of Spatio-temporal consistency
    - utils
    - feature_extract.py
    - main.py
    - trainer.py
    - option.py
## Dataset

Prepare dataset in txt file, and each row of txt is as below:
The split of hmdb51/ucf101/kinetics-400 can be download from 
[google driver](https://drive.google.com/drive/folders/1gOOOlBJtH1p2weC-k7TD3HcC2HSHBryw?usp=sharing).

Each item include
> video_path class frames_num

## VTDL
### Network Architecture
The network is in the folder **src/model/[backbone].py**

|  Method   | #logits_channel  |
|  ----  | ----  | 
| C3D  | 512 |
| R2P1D  | 2048 |
| I3D | 1024 |
| R3D  | 2048 |


### Step1: self-supervised learning
#### HMDB51
```bash
bash scripts/TemporalDisc/hmdb51.sh
```
#### UCF101
```bash
bash scripts/TemporalDisc/ucf101.sh
```
#### Kinetics-400
```bash
bash scripts/TemporalDisc/kinetics.sh
```

**Notice: More Training Options and ablation study Can be find in scripts**

### Step2: Transfer to action recognition
#### HMDB51
```bash
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
--workers 8 \
--dropout 0.5 \
--gpus 2 \
--logs_path ../experiments/logs/hmdb51_i3d_ft \
--print-freq 100 \
--weights ../experiments/TemporalDis/hmdb51/models/04-16-2328_aug_CJ/ckpt_epoch_48.pth
```
#### UCF101
```bash
#!/usr/bin/env bash
python main.py \
--method ft \
--train_list ../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt \
--val_list ../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--arch i3d \
--mode rgb \
--lr 0.0005 \
--lr_steps 10 20 25 30 35 40 \
--epochs 45 \
--batch_size 4 \
--data_length 64 \
--workers 8 \
--dropout 0.5 \
--gpus 2 \
--logs_path ../experiments/logs/ucf101_i3d_ft \
--print-freq 100 \
--weights ../experiments/TemporalDis/ucf101/models/04-18-2208_aug_CJ/ckpt_epoch_45.pth
```
**Notice: More Training Options and ablation study Can be find in scripts**

## Results
### Step2:Transfer

With same experiment setting, the result is reported below:

|  Method   | UCF101  | HMDB51 |
|  ----  | ----  | ---- |
| Baseline  | 60.3 | 22.6| 
| + BA |63.3 | 26.2|
| + Temporal Discriminative  | 72.7 | 41.2| 
| + TCA | 82.3 |52.9|

#### trained models/logs/performance

We provided trained models/logs/performance in google driver.
##### Baseline + BA

![BA_fine_tune_performance.png](https://i.loli.net/2020/07/04/V1YzdPhxjJARnKS.png)

[performance](https://drive.google.com/file/d/13O7JpIGYxspgOsJTCKLh37ZeKR2slPgz/view?usp=sharing);

[trained_model](https://drive.google.com/file/d/10J5fKKkDF58njdsLGXkknk0RWkaINJ31/view?usp=sharing); 

[logs](https://drive.google.com/file/d/1K1_U692NZq5F53DDPDkpw7JFiEqIqole/view?usp=sharing)

##### Baseline + BA + Temporal Discriminative

![wo_TCA_fine_tune_performance.png](https://i.loli.net/2020/07/04/ZYkEIRcMqxzD6AX.png)

[performance](https://drive.google.com/file/d/1ZHUhFsHoyyIWTnoDB1XEG2WBJLgxTtA9/view?usp=sharing);

[trained_model](https://drive.google.com/file/d/1HJQwzRwNs5nOseAnPW87iDpUMXOs78a8/view?usp=sharing);

[logs](https://drive.google.com/file/d/1Q-tq9bf-J8caHxJJDMlXo4KX5cNJ_75w/view?usp=sharing)

##### Baseline + BA + Temporal Discriminative + TCA

**(a). Pretrain**

Loss curve:

![loss.png](https://i.loli.net/2020/09/18/4dMhnxJtjupE1QH.png)

Ins Prob:

![prob.png](https://i.loli.net/2020/09/18/4QIj5PZR1pi3rXn.png)

[pretrained_weight](https://drive.google.com/file/d/1g1reGkcD2xwztzwfGRPLzz0JMOP3yo7a/view?usp=sharing)

This pretrained model can achieve 52.7% on HMDB51.

**(b). Finetune**


![VTDL_fine_tune_performance.png](https://i.loli.net/2020/07/04/14TlPxKcvOyLguM.png)

[performance](https://drive.google.com/file/d/16GL88PLLOpLoO_yWjK2XcYPKKnqRBtcr/view?usp=sharing);

[trained_model](https://drive.google.com/file/d/1TdxIBrdLcgKabL5A_FusomWLe4oXDcDK/view?usp=sharing);

[logs](https://drive.google.com/file/d/13vfdQusv2Gd42nYXe4p1nWEByIJ9cLoz/view?usp=sharing)


The result is report with single video clip. In the test, we will average ten clips as final predictions. Will lead to around 2-3% improvement.
```bash
python test.py
```


## Feature Extractor
As STCR can be easily extend to other video representation task, we offer the scripts to perform feature extract.
```bash
python feature_extractor.py
```


The feature will be saved as a single numpy file in the format [video_nums,features_dim]


## Citation

Please cite our paper if you find this code useful for your research.

```
@Article{wang2020self,
  author  = {Jinpeng Wang and Yiqi Lin and Andy J. Ma and Pong C. Yuen},
  title   = {Self-supervised Temporal Discriminative Learning for Video Representation Learning},
  journal = {arXiv preprint arXiv:2008.02129},
  year    = {2020},
}
```


## Others

The project is partly based on [Unsupervised Embedding Learning](https://github.com/mangye16/Unsupervised_Embedding_Learning) and [MOCO](https://github.com/facebookresearch/moco).