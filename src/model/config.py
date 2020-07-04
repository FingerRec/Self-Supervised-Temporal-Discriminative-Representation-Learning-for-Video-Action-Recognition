from model.i3d import I3D
from model.r2p1d import R2Plus1DNet
from model.r3d import resnet18
from model.c3d import C3D
import torch.nn as nn
from model.model import TCN
import torch
import torch.backends.cudnn as cudnn
from utils.load_weights import load_weight


def model_config(args, num_class):
    if args.eval_indict == 'acc':
        with_classifier = True
    else:
        with_classifier = False
    if args.arch == 'i3d':
        base_model = I3D(num_classes=num_class, modality=args.mode, dropout_prob=args.dropout, with_classifier=with_classifier)
        args.logits_channel = 1024
        if args.spatial_size == '112':
            out_size = (int(args.data_length) // 8, 4, 4)
        else:
            out_size = (int(args.data_length) // 8, 7, 7)
    elif args.arch == 'r2p1d':
        base_model = R2Plus1DNet((2, 2, 2, 2), num_classes=num_class, with_classifier=with_classifier)
        args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'c3d':
        base_model = C3D(num_classes=num_class, with_classifier=with_classifier)
        args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d':
        base_model = resnet18(num_classes=num_class, with_classifier=with_classifier)
        args.logits_channel = 512
        out_size = (4, 4, 4)
    else:
        Exception("unsuporrted arch!")
    if args.mutual_learning == 1 and args.eval_indict == 'loss':
        base_model_2 = type(base_model)(num_classes=num_class, with_classifier=with_classifier)
        base_model_2.load_state_dict(base_model.state_dict())
        base_models = nn.Sequential(base_model, base_model_2)
        # for i in range(args.mutual_num):
        #     base_models.append(load_weight(args, base_model))
        model = TCN(base_models, out_size, args)
        #load_weight(args, base_model)
        model = load_weight(args, model)
        model = nn.DataParallel(model).cuda()
    elif args.eval_indict == 'loss' and args.pt_loss == 'TemporalDis':
        base_model = load_weight(args, base_model)
        base_model_2 = type(base_model)(num_classes=num_class, with_classifier=with_classifier)
        base_model_2.load_state_dict(base_model.state_dict())
        base_models = nn.Sequential(base_model, base_model_2)
        # for i in range(args.mutual_num):
        #     base_models.append(load_weight(args, base_model))
        model = TCN(base_models, out_size, args)
        # load_weight(args, base_model)
        # model = load_weight(args, model)
        model = nn.DataParallel(model).cuda()
    else:
        base_model = load_weight(args, base_model)
        model = TCN(base_model,  out_size, args)
        model = nn.DataParallel(model).cuda()
    # cudnn.benchmark = True
    return model
