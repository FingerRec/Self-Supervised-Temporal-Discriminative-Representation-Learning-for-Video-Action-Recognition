"""
Training TemporalDis and Instance Discrimination
InsDis: Unsupervised feature learning via non-parametric instance discrimination
TemporalDis: Momentum Contrast for Unsupervised Visual Representation Learning
"""
from __future__ import print_function

import numpy as np
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger

from utils.utils import  AverageMeter

from model.i3d import I3D
from model.r2p1d import R2Plus1DNet
from NCE.NCEAverage import MemoryInsDis
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from data.config import data_config, augmentation_config
from data.dataloader import data_loader_init
from utils.load_weights import weights_init
import datetime
from utils.utils import accuracy
import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#==
from TC.gen_positive import GenPositive
from TC.gen_negative import GenNegative
try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    # if epoch < 2:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-7
    #     return 0
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=3, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--dataset', type=str, default='hmdb51', choices=['hmdb51', 'ucf101', 'kinetics'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # augmentation setting
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])

    # warm up
    parser.add_argument('--warm', action='store_true', help='add warm-up setting')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--arch', default='i3d', type=str, choices=['i3d', 'r3d', 'r2p1d', 'c3d'])

    # loss function
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # memory setting
    parser.add_argument('--moco', action='store_true', help='using TemporalDis (otherwise Instance Discrimination)')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # dataset setting
    parser.add_argument('--train_list', type=str, default='../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt')
    parser.add_argument('--val_list', type=str, default='../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt')

    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('amax'):
        opt.data_folder = '../experiments/TemporalDis/{}/'.format(opt.dataset)
        opt.model_path = '../experiments/TemporalDis/{}/models'.format(opt.dataset)
        opt.tb_path = '../experiments/TemporalDis/{}/tensorboard'.format(opt.dataset)
        opt.tb_path2 = '../experiments/TemporalDis/{}/tensorboard2'.format(opt.dataset)
        opt.kmeans_path = '../experiments/TemporalDis/{}/k_means'.format(opt.dataset)
        opt.pseduo_model_path = '../experiments/TemporalDis/{}/pseudo_models'.format(opt.dataset)
        opt.tsne_path = '../experiments/TemporalDis/{}/tsne'.format(opt.dataset)
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))


    # opt.dataset = 'ucf101'
    # opt.dataset = 'kinetics'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    prefix = 'TemporalDis{}'.format(opt.alpha) if opt.moco else 'InsDis'
    date = datetime.datetime.today().strftime('%m-%d-%H%M')
    opt.model_name = date
    # opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_crop_{}'.format(prefix, opt.method, opt.nce_k, opt.model,
    #                                                                     opt.learning_rate, opt.weight_decay,
    #                                                                     opt.batch_size, opt.crop)

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.tb_folder2 = os.path.join(opt.tb_path2, opt.model_name)
    if not os.path.isdir(opt.tb_folder2):
        os.makedirs(opt.tb_folder2)

    opt.kmeans_folder = os.path.join(opt.kmeans_path, opt.model_name)
    if not os.path.isdir(opt.kmeans_folder):
        os.makedirs(opt.kmeans_folder)

    opt.pseduo_model_folder = os.path.join(opt.pseduo_model_path, opt.model_name)
    if not os.path.isdir(opt.pseduo_model_folder):
        os.makedirs(opt.pseduo_model_folder)

    opt.tsne_folder = os.path.join(opt.tsne_path, opt.model_name)
    if not os.path.isdir(opt.tsne_folder):
        os.makedirs(opt.tsne_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)
        # p2.data.mul_(m).add_(1 - m, p1.data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def saving(logger, loss, epoch, optimizer, args, model, contrast, prob, model_ema, tag='TemporalDis'):
    if tag == 'TemporalDis':
        model_folder = args.model_folder
    elif tag == 'Pseudo':
        model_folder = args.pseduo_model_folder
    else:
        Exception("not implement")
    # tensorboard logger
    logger.log_value('ins_loss', loss, epoch)
    logger.log_value('ins_prob', prob, epoch)
    logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # save model
    if epoch % args.save_freq == 0:
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if args.moco:
            state['model_ema'] = model_ema.state_dict()
        if args.amp:
            state['amp'] = amp.state_dict()
        save_file = os.path.join(model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        torch.save(state, save_file)
        # help release GPU memory
        del state

    # saving the model
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if args.moco:
        state['model_ema'] = model_ema.state_dict()
    if args.amp:
        state['amp'] = amp.state_dict()
    save_file = os.path.join(model_folder, 'current.pth')
    torch.save(state, save_file)
    if epoch % args.save_freq == 0:
        save_file = os.path.join(model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        torch.save(state, save_file)
    # help release GPU memory
    del state
    torch.cuda.empty_cache()

def main():

    args = parse_option()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model and optimizer
    # == dataset config==
    """
    CUDA_VISIBLE_DEVICES=0,1 python train_temporal_dis.py \
     --batch_size 16 --num_workers 8 --nce_k 3569 --softmax --moco
    """

    # args.dataset = 'hmdb51'
    # args.train_list = '../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt'
    # args.val_list = '../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt'
    """
    CUDA_VISIBLE_DEVICES=1 python train_temporal_dis.py \
     --batch_size 16 --num_workers 8 --nce_k 9536 --softmax --moco
    """
    # args.print_freq = 100
    # args.dataset = 'ucf101'
    # args.train_list = '../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt'
    # args.val_list = '../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt'

    # args.print_freq = 1000
    # args.dataset = 'kinetics'
    # args.train_list = '../datasets/lists/kinetics-400/ssd_kinetics_video_trainlist.txt'
    # args.val_list = '../datasets/lists/kinetics-400/ssd_kinetics_video_vallist.txt'

    args.dropout = 0.5
    args.clips = 1
    args.data_length = 16
    args.stride = 4
    args.spatial_size = 224
    args.root = ""
    args.mode = 'rgb'
    args.eval_indict = 'loss'
    args.pt_loss = 'TemporalDis'
    args.workers = 4
    # args.arch = 'i3d' # 'r2p1d'
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms, eval_transforms = augmentation_config(args)
    train_loader, val_loader, eval_loader, train_samples, val_samples, eval_samples = data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms)

    n_data = len(train_loader)
    if args.arch == 'i3d':
        model = I3D(num_classes=101, modality=args.mode, dropout_prob=args.dropout, with_classifier=False)
        model_ema = I3D(num_classes=101, modality=args.mode, dropout_prob=args.dropout, with_classifier=False)
    elif args.arch == 'r2p1d':
        model = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
        model_ema = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
    elif args.arch == 'r3d':
        from model.r3d import resnet18
        model = resnet18(num_classes=num_class, with_classifier=False)
        model_ema = resnet18(num_classes=num_class, with_classifier=False)
    else:
        Exception("Not implemene error!")
    model = torch.nn.DataParallel(model)
    model_ema = torch.nn.DataParallel(model_ema)
    # random initialization
    model.apply(weights_init)
    model_ema.apply(weights_init)
    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    contrast = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, args.softmax).cuda(args.gpu)
    # contrast2 = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, args.softmax).cuda(args.gpu)
    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion = criterion.cuda(args.gpu)
    cls_criterion = nn.CrossEntropyLoss().cuda()

    model = model.cuda()
    if args.moco:
        model_ema = model_ema.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        if args.moco:
            optimizer_ema = torch.optim.SGD(model_ema.parameters(),
                                            lr=0,
                                            momentum=0,
                                            weight_decay=0)
            model_ema, optimizer_ema = amp.initialize(model_ema, optimizer_ema, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.moco:
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    logger2 = tb_logger.Logger(logdir=args.tb_folder2, flush_secs=2)

    #==================================== our data augmentation method=================================
    pos_aug = GenPositive()
    neg_aug = GenNegative()

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args, pos_aug, neg_aug)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        saving(logger, loss, epoch, optimizer, args, model, contrast, prob, model_ema, 'TemporalDis')

        #================iterative update ================================
        # pseudo_labels, tsne_features = generate_pseudo_label(model, train_loader, train_samples, epoch, args)
        # # plot_tsne(tsne_features, epoch, args, num_class, 'TemporalDis')
        # pse_cls_loss = train_pseduo_label(train_loader, model, model_ema, cls_criterion, optimizer, args, epoch, pseudo_labels)
        # print("pse_cls_loss:{}".format(pse_cls_loss))
        # # plot_tsne(tsne_features, epoch, args, num_class, 'Pseudo')
        # saving(logger2, pse_cls_loss, epoch, optimizer, args, model, contrast, prob, model_ema, 'Pseudo')
        # if epoch % 5 == 0:
        #     pseudo_labels, tsne_features = generate_pseudo_label(model_ema, train_loader, train_samples, epoch, args)
        #     # plot_tsne(tsne_features, epoch, args, num_class, 'TemporalDis')
        #     for j in range(3):
        #         # pseudo_labels, tsne_features = generate_pseudo_label(model, train_loader, train_samples, epoch, args)
        #         # plot_tsne(tsne_features, epoch, args, num_class, 'TemporalDis')
        #         pse_cls_loss = train_pseduo_label(train_loader, model, model_ema, cls_criterion, optimizer, args, epoch, pseudo_labels)
        #         print("pse_cls_loss:{}".format(pse_cls_loss))
        #         # plot_tsne(tsne_features, epoch, args, num_class, 'Pseudo')
        #         saving(logger2, pse_cls_loss, epoch, optimizer, args, model, contrast, prob, model_ema, 'Pseudo')


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt, pos_aug, neg_aug):
    """
    one epoch training for instance discrimination
    """
    print("==> (TemporalDis) training...")
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs[0].size(0)
        # fixed args.batch_size
        if bsz < opt.batch_size:
            print("batch less than 16, continue")
            continue
        inputs[0] = inputs[0].float()
        inputs[1] = inputs[1].float()
        if opt.gpu is not None:
            inputs[0] = inputs[0].cuda(opt.gpu, non_blocking=True)
            inputs[1] = inputs[1].cuda(opt.gpu, non_blocking=True)
        else:
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        anchor, positive, negative =  inputs

        # here a series of data augmentation
        # ====================================================postive operation=======================
        anchor = pos_aug(anchor)
        # negative = neg_aug(negative)
        # strong_negative = neg_aug(positive)

        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        # # data rotation
        # rotation_data_1 = []
        # flip_labels = torch.ones(x1.size(0) * 4)
        # for i in range(4):
        #     rotation_data_1.append(four_rotation_cls(x1, torch.ones(x1.size(0)) * i))
        #     flip_labels[x1.size(0) * i:x1.size(0) * (i + 1)] = torch.ones(
        #         x1.size(0)) * i
        # x1 = torch.cat(rotation_data_1, dim=0)
        # x1_rotate_label = torch.LongTensor(flip_labels.long()).cuda()
        # rotation_data_2 = []
        # flip_labels = torch.ones(x2.size(0) * 4)
        # for i in range(4):
        #     rotation_data_2.append(four_rotation_cls(x2, torch.ones(x2.size(0)) * i))
        #     flip_labels[x2.size(0) * i:x2.size(0) * (i + 1)] = torch.ones(
        #         x2.size(0)) * i
        # x2 = torch.cat(rotation_data_2, dim=0)
        # x2_rotate_label = torch.LongTensor(flip_labels.long()).cuda()

        feat_q, cls_q, mix_q = model(anchor)
        with torch.no_grad():
            positive = positive[shuffle_ids]
            feat_k, cls_k, mix_k = model_ema(positive)
            feat_k = feat_k[reverse_ids]
        feat_n, cls_n, mix_n = model(negative)
        # feat_sn, cls_sn, mix_sn = model(strong_negative)
        out = contrast(feat_q, feat_k, feat_n, index)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        sample_loss = triplet_loss(feat_q, feat_k, feat_n)
        # out = contrast(feat_q, feat_k, feat_n, index)
        # out = contrast(feat_q, feat_k, index)
        contrast_loss = criterion(out)
        # out2 = contrast2(feat_n, feat_k, _, index)
        # contrast_loss2 = criterion(out2)
        # print(contrast_loss, contrast_loss2)
        # print(contrast_loss, sample_loss)
        loss = contrast_loss # + sample_loss # + contrast_loss2 # + cls_loss + mixup_loss
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        message = ('TemporalDis Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(message)
            # print(out.shape)
            sys.stdout.flush()
            with open("../experiments/MoCo_kinetics.txt", 'a') as f:
                f.write(message + '\n')
    return loss_meter.avg, prob_meter.avg

def generate_pseudo_label(model, train_loader, train_samples, epoch, args, feature_dim=2048, clusters=200):
    """
    generate pseduo label for all dataset
    :return:
    """
    #step 1: feature extractor....
    #step 2: apply k_means algorithm and save pseudo label in an array
    #step 3: plot pseudo label
    # ==============================================Step 1 ===============================
    print("==> (generate pseudo label): feature extractor")
    model.eval()
    features = np.random.rand(train_samples, feature_dim) # the number of training samples
    tsne_features = {'data': [], 'target': []}
    with torch.no_grad():
        for i, (input, target, index) in enumerate(train_loader):
            _, inputs, _ = input
            inputs = inputs.cuda()
            _, _, feat = model(inputs)
            for j in range(feat.size(0)):
                features[index[j]] = feat[j].data.cpu().numpy()
            for j in range(feat.size(0)):
                tsne_features['data'].append(feat[j])
                tsne_features['target'].append(target[j])
            if i % args.print_freq == 0:
                print("epoch: {}, {}/{} finished feature extract".format(epoch, i, len(train_loader)))
            # if i > 50:
            #     break
    # ==============================================Step 2 clustering =================================
    from PSEUDO.clustering import Kmeans
    print("==> (generate pseudo label) k-means cluster")
    cluster = Kmeans(clusters)
    cluster.cluster(features)
    labels = cluster.images_lists #list, len=clusters, each cluster include index
    pseudo_labels = np.zeros(train_samples)
    for i in range(len(labels)):
        for item in labels[i]:
            pseudo_labels[item] = i
    pseudo_labels = torch.tensor(pseudo_labels).cuda().long()
    print(pseudo_labels)
    # from sklearn.cluster import KMeans
    # from sklearn.decomposition import PCA
    # reduced_data = PCA(n_components=10).fit_transform(features)
    # k_means = KMeans(init='k-means++', n_clusters=clusters, n_init=1000)
    # k_means.fit(reduced_data)
    # print(k_means.labels_)
    # labels = torch.tensor(k_means.labels_).cuda().long()
    # return labels
    # =============================================Step 3 visualize =================================
    # print("==> (generate pseudo label) plot and save k-means cluster")
    # import matplotlib.pyplot as plt
    # # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    # # Plot the decision boundary. For that, we will assign a color to each
    # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')
    #
    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = k_means.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # plt.title('K-means clustering on the UCF101 dataset (PCA-reduced data)\n'
    #           'Centroids are marked with white cross')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.savefig("{}/{}.png".format(args.kmeans_folder, epoch))
    # plt.show()
    return pseudo_labels, tsne_features

def plot_tsne(data, epoch, args, num_class, front='TemporalDis'):
    """
    plot the tsne visualization result and record it in a file
    :return:
    """
    print("==> (generate pseudo label) t-sne visualization")
    from utils.visualization.t_SNE_Visualization import tsne_visualize
    front = front + '_' + str(epoch)
    file_name = "{}/{}.png".format(args.tsne_folder, front)
    tsne_visualize(data, file_name, num_class)
    return True

def train_pseduo_label(train_loader, model, model_ema, criterion, optimizer, opt, epoch, pseduo_labels):
    """
    :return:
    """
    # need clean data and labels
    print("==> train pseduo label")
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (videos, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        _, inputs, _ = videos
        bsz = inputs.size(0)
        inputs = inputs.cuda()
        target = pseduo_labels[index].cuda()
        # ===================forward=====================
        _, predict, _ = model(inputs)
        loss = criterion(predict, target)
        prec1, prec5 = accuracy(predict.data, target, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        # moment_update(model, model_ema, opt.alpha)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Pseudo Label Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Top1 {top1.val:3f} ({top1.avg:.3f})\t'
                  'Top5 {top5.val:3f} ({top5.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter, top1=top1,top5=top5))
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()