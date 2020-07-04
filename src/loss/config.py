import torch
import torch.optim as optim
from loss.inverse_loss import inverse_loss, list_inverse_loss, inverse_cls_loss
# from loss.batch_flip_cls_loss import BatchFlipLoss, BatchFlipValLoss
from loss.batch_NCE import BatchFlipLoss, BatchFlipValLoss
from loss.triplet_loss import TripletLoss
from loss.temporal_consistency_loss import TemporalConsistencyTrainLoss, TemporalConsistencyValLoss
from loss.net_mixup_loss import mixup_loss
from loss.mutual_loss import mutual_loss
from loss.batch_NCE import BatchCriterion
from loss.contrastiveLoss import ContrastiveLoss
from loss.temporal_sampling_rate_loss import tsc_loss
from loss.MoCo_loss import MoCoLoss


def optim_init(args, model):
    if args.eval_indict == 'acc':
        train_criterion = torch.nn.NLLLoss().cuda()
        val_criterion = torch.nn.NLLLoss().cuda()
    elif args.eval_indict == 'loss':
        if args.pt_loss == 'flip':
            train_criterion = inverse_loss
            val_criterion = inverse_loss
            # criterion = list_inverse_loss
        elif args.pt_loss == 'triplet':
            train_criterion = TripletLoss()
            val_criterion = TripletLoss()
            # criterion = list_inverse_loss
        elif args.pt_loss == 'flip_cls':
            # train_criterion = BatchFlipLoss(nce=args.nce, flip_classes=8, batch_size=args.batch_size)
            train_criterion = BatchFlipLoss(nce=args.nce, flip_classes=8, batch_size=args.batch_size)
            val_criterion = BatchFlipValLoss()
        elif args.pt_loss == 'temporal_consistency':
            train_criterion = TemporalConsistencyTrainLoss(nce=args.nce, flip_classes=16, batch_size=args.batch_size)
            val_criterion = TemporalConsistencyValLoss()
        elif args.pt_loss == 'net_mixup':
            train_criterion = mixup_loss
            val_criterion = mixup_loss
        elif args.pt_loss == 'mutual_loss':
            train_criterion = mutual_loss
            val_criterion = mutual_loss
        elif args.pt_loss == 'instance_discriminative':
            # train_criterion = ContrastiveLoss(0)
            # val_criterion = ContrastiveLoss(0)
            # train_criterion = BatchCriterion(1, 0.1, args.batch_size).cuda()
            # val_criterion = BatchCriterion(1, 0.1, args.batch_size).cuda()
            train_criterion = ContrastiveLoss(loss_type=1,
                                              batch_size=args.batch_size,
                                              tempeature=1,
                                              num_channels=256)
            val_criterion = ContrastiveLoss(loss_type=1,
                                              batch_size=args.batch_size,
                                              num_channels=256,
                                            tempeature=1,
                                            test=True)
        elif args.pt_loss == 'TSC':
            train_criterion = torch.nn.CrossEntropyLoss().cuda()
            val_criterion = torch.nn.CrossEntropyLoss().cuda()
            # train_criterion = tsc_loss
            # val_criterion = tsc_loss
        elif args.pt_loss == 'TemporalDis':
            train_criterion = MoCoLoss(test=False, queue_size=9000, num_channels=1024).cuda()
            val_criterion = MoCoLoss(test=True, queue_size=3000, num_channels=1024).cuda()
        else:
            Exception("unsupported loss now!")
    else:
        Exception("wrong optim init!")
    if args.eval_indict == 'loss' and args.pt_loss == 'TemporalDis':
        parameters = model.module.base_model[0].parameters()
    else:
        parameters = model.parameters()
    if args.optim == 'sgd':
        optimizer = optim.SGD(parameters,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr)
    else:
        Exception("not supported optim")
    return train_criterion, val_criterion, optimizer

