import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from data.config import data_config, augmentation_config
from data.dataloader import data_loader_init
from model.config import model_config
from TC.config import TC
from option import args
import torch.nn as nn

def single_extract(tc, val_loader, model):
    model.eval()
    features = {'data':[], 'target':[]}
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            inputs = tc(input)
            output = model(inputs)
            output = nn.AdaptiveAvgPool3d(1)(output).view(output.size(0), output.size(1))
            # print(output.size())
            # print(target)
            for j in range(output.size(0)):
                features['data'].append(output[j])
                features['target'].append(target[j])
            if i%10 == 0:
                print("{}/{} finished".format(i, len(val_loader)))
    return features

def feature_extract(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cudnn.benchmark = True
    # == dataset config==
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms, _ = augmentation_config(args)
    train_data_loader, val_data_loader, _, _, _, _ = data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, _)
    # == model config==
    model = model_config(args, num_class)
    tc = TC(args)
    features = single_extract(tc,val_data_loader, model)
    return features

def main():
    features = feature_extract(args)
    np.save('../experiments/visualization/TSNE/self_supervised_UCF101MoCo_hmdb51_finetune_features.npy', features)


if __name__ == '__main__':
    main()