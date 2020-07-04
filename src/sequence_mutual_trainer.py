#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from utils.utils import Timer
import torch
import torch.backends.cudnn as cudnn
from utils.utils import AverageMeter
from data.config import data_config, augmentation_config
from data.dataloader import data_loader_init
from model.config import model_config
from loss.config import optim_init
from utils.learning_rate_adjust import adjust_learning_rate
from utils.checkpoint_record import Record
from TC.config import TC
from utils.utils import accuracy
from utils.gradient_check import plot_grad_flow
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

lowest_val_loss = float('inf')
best_prec1 = 0
torch.manual_seed(1)


def train(args, tc, train_loader, models, criterion, optimizers, epoch, recorder):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []
    top1 = []
    top3 = []
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    loss_ce = nn.CrossEntropyLoss()
    for i in range(args.mutual_num):
        models[i].train()
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        top3.append(AverageMeter())
    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda()
        inputs = tc(input)
        outputs = []
        for model in models:
            outputs.append(model(inputs))
        for i in range(args.mutual_num):
            ce_loss = loss_ce(outputs[i], inputs[2])
            kl_loss = 0
            for j in range(args.mutual_num):
                if i != j:
                    kl_loss += loss_kl(F.log_softmax(outputs[i], dim=1),
                                            F.softmax(Variable(outputs[j]), dim=1))
            loss = ce_loss + kl_loss / (args.mutual_num - 1)

            # measure accuracy and record loss
            prec = accuracy(outputs[i].data, inputs[2].data, topk=(1,))[0]
            losses[i].update(loss.item(), input.size()[0])

            # compute gradients and update SGD
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=optimizers[0].param_groups[-1]['lr']))
            print(message)
            recorder.record_message('a', message)
            if args.eval_indict == 'acc':
                message = "Training: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                print(message)
                recorder.record_message('a', message)
            else:
                if args.pt_loss == 'flip_cls' or 'temporal_consistency' or 'mutual_loss':
                    message = "Training: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                    print(message)
                    recorder.record_message('a', message)
    if args.eval_indict == 'acc':
        return top1.avg, losses.avg
    else:
        return losses.avg


def validate(args, tc, val_loader, models, criterion, recorder):
    batch_time = AverageMeter()
    losses = []
    top1 = []
    top3 = []
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    loss_ce = nn.CrossEntropyLoss()
    # switch to evaluate mode
    for i in range(args.mutual_num):
        models[i].eval()
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        top3.append(AverageMeter())
    end = time.time()
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            target = target.cuda()
            inputs = tc(input)
            outputs = []
            for model in models:
                outputs.append(model(inputs))
            for i in range(args.mutual_num):
                ce_loss = loss_ce(outputs[i], inputs[2])
                kl_loss = 0
                for j in range(args.mutual_num):
                    if i != j:
                        kl_loss += loss_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(Variable(outputs[j]), dim=1))
                loss = ce_loss + kl_loss / (args.mutual_num - 1)

                # measure accuracy and record loss`
                prec = accuracy(outputs[i].data, inputs[2].data, topk=(1,))[0]
                losses[i].update(loss.item(), input.size()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                message = ('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses
                    ))
                print(message)
                recorder.record_message('a', message)
                if args.eval_indict == 'acc':
                    message = "Eval: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                    print(message)
                    recorder.record_message('a', message)
                else:
                    if args.pt_loss == 'temporal_consistency' or 'flip_cls' or 'mutual_loss':
                        message = "Val: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                        print(message)
                        recorder.record_message('a', message)
    if args.eval_indict == 'acc':
        return top1.avg, losses.avg
    else:
        return losses.avg


def train_and_eval(args):
    # =
    global lowest_val_loss, best_prec1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(1)
    cudnn.benchmark = True
    timer = Timer()
    recorder = Record(args)
    # == dataset config==
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms = augmentation_config(args)
    train_data_loader, val_data_loader = data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms)
    # == model config==
    models = []
    optimizers = []
    for i in range(args.mutual_num):
        model = model_config(args, num_class)
        models.append(model)
    recorder.record_message('a', '='*100)
    recorder.record_message('a', str(model.module))
    recorder.record_message('a', '='*100)
    # == optim config==
    for i in range(args.mutual_num):
        train_criterion, val_criterion, optimizer = optim_init(args, model)
        optimizers.append(optimizer)
    # == data augmentation(self-supervised) config==
    tc = TC(args)
    # == train and eval==
    for epoch in range(args.start_epoch, args.epochs):
        timer.tic()
        for i in range(args.mutual_num):
            adjust_learning_rate(optimizers[i], args.lr, epoch, args.lr_steps)
        if args.eval_indict == 'acc':
            train_prec1, train_loss = train(args, tc, train_data_loader, models, train_criterion, optimizers, epoch, recorder)
            # train_prec1, train_loss = random.random() * 100, random.random()
            recorder.record_train(train_loss / 5.0, train_prec1 / 100.0)
        else:
            train_loss = train(args, tc, train_data_loader, models, train_criterion, optimizers, epoch, recorder)
            # train_prec1, train_loss = random.random() * 100, random.random()
            recorder.record_train(train_loss)
        if (epoch + 1) % args.eval_freq == 0:
            if args.eval_indict == 'acc':
                val_prec1, val_loss = validate(args, tc, val_data_loader, models, val_criterion, recorder)
                # val_prec1, val_loss = random.random() * 100, random.random()
                recorder.record_val(val_loss / 5.0, val_prec1 / 100.0)
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)
                checkpoint = {'epoch': epoch + 1, 'arch': "i3d", 'state_dict': model.state_dict(),
                              'best_prec1': best_prec1}
            else:
                val_loss = validate(args, tc, val_data_loader, models, val_criterion, recorder)
                # val_loss = random.random()
                # val_prec1, val_loss = random.random() * 100, random.random()
                recorder.record_val(val_loss)
                is_best = val_loss < lowest_val_loss
                lowest_val_loss = min(val_loss, lowest_val_loss)
                checkpoint = {'epoch': epoch + 1, 'arch': "i3d", 'state_dict': model.state_dict(), 'lowest_val': lowest_val_loss}
        recorder.save_model(checkpoint,  is_best)
        timer.toc()
        left_time = timer.average_time * (args.epochs - epoch)

        if args.eval_indict == 'acc':
            message = "best_prec1 is: {} left time is : {}".format(best_prec1, timer.format(left_time))
        else:
            message = "lowest_val_loss is: {} left time is : {}".format(lowest_val_loss, timer.format(left_time))
        print(message)
        recorder.record_message('a', message)
    # return recorder.best_name
    return recorder.filename

