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
from utils.moment_update import update_ema_variables
from utils.gradient_check import plot_grad_flow
import random

lowest_val_loss = float('inf')
best_prec1 = 0
torch.manual_seed(1)

def train(args, tc, train_loader, model, criterion, optimizer, epoch, recorder, MoCo_init=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    if MoCo_init:
        model.eval()
    else:
        model.train()
    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        index = index.cuda(async =True)
        inputs = tc(input)
        target = torch.autograd.Variable(target)
        # inputs = torch.autograd.Variable(inputs)
        output = model(inputs)
        # print(index)
        # print(index.size(), output.size())
        if args.eval_indict == 'acc':
            loss = criterion(output, target)
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
        else:
            if args.pt_loss == "TSC":
                index = index.view(output.size(0))
            # print(output.size(), index.size())
            if args.pt_loss == 'TemporalDis' and MoCo_init:
                print("train set {}/{} finished initialization".format(i, int(2000/args.batch_size)))
                if i > int(2000/args.batch_size):
                    break
                criterion(output, index, True)
                continue
            loss = criterion(output, index)
            losses.update(loss.data.item(), input[0].size(0))
            if args.pt_loss == 'flip_cls':
                prec1, prec3 = accuracy(output[0].data, output[1].data, topk=(1, 3))
                top1.update(prec1.item(), input[0].size(0))
                top3.update(prec3.item(), input[0].size(0))
            elif args.pt_loss == 'mutual_loss':
                prec1, prec3 = accuracy(output[0].data, output[2].data, topk=(1, 3))
                top1.update(prec1.item(), input[0].size(0))
                top3.update(prec3.item(), input[0].size(0))
            elif args.pt_loss == 'temporal_consistency':
                # print(output[0].size(), output[1].size(), output[4].size())
                prec1, prec3 = accuracy(output[0].data, output[4].data, topk=(1, 3))
                top1.update(prec1.item(), input[0].size(0))
                top3.update(prec3.item(), input[0].size(0))
            elif args.pt_loss == 'TSC':
                prec1, prec3 = accuracy(output.data, index, topk=(1, 3))
                top1.update(prec1.item(), output.size(0))
                top3.update(prec3.item(), output.size(0))
        # if args.eval_
        if args.eval_indict == 'loss' and args.pt_loss == 'TemporalDis':
            update_ema_variables(model.module.base_model[1], model.module.base_model[0], 0.99)
        optimizer.zero_grad()
        loss.backward()
        # # gradient check
        # plot_grad_flow(model.module.base_model.named_parameters())
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr']))
            print(message)
            recorder.record_message('a', message)
            if args.eval_indict == 'acc':
                message = "Finetune Training: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                print(message)
                recorder.record_message('a', message)
            else:
                for name in ('flip_cls', 'temporal_consistency', 'mutual_loss', 'TSC'):
                    if args.pt_loss == name:
                        message = "Self-supervised Training: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                        print(message)
                        recorder.record_message('a', message)
    if args.eval_indict == 'acc':
        return top1.avg, losses.avg
    else:
        return losses.avg


def validate(args, tc, val_loader, model, criterion, recorder, MoCo_init=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            target = target.cuda(async=True)
            inputs = tc(input)
            # inputs = torch.autograd.Variable(inputs)
            target = torch.autograd.Variable(target)
            index = index.cuda(async=True)
            output = model(inputs)
            if args.eval_indict == 'acc':
                loss = criterion(output, target)
                prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top3.update(prec3.item(), input.size(0))
            else:
                if args.pt_loss == "TSC":
                    index = index.view(output.size(0))
                if args.pt_loss == 'TemporalDis' and MoCo_init:
                    print("val set {}/{} finished initialization".format(i, int(2000/args.batch_size)))
                    if i > int(2000/args.batch_size):
                        break
                    criterion(output, index, True)
                    continue
                loss = criterion(output, index)
                losses.update(loss.data.item(), input[0].size(0))
                if args.pt_loss == 'flip_cls':
                    prec1, prec3 = accuracy(output[0].data, output[1].data, topk=(1, 3))
                    top1.update(prec1.item(), input[0].size(0))
                    top3.update(prec3.item(), input[0].size(0))
                elif args.pt_loss == 'mutual_loss':
                    prec1, prec3 = accuracy(output[0].data, output[2].data, topk=(1, 3))
                    top1.update(prec1.item(), input[0].size(0))
                    top3.update(prec3.item(), input[0].size(0))
                elif args.pt_loss == 'temporal_consistency':
                    prec1, prec3 = accuracy(output[0].data, output[4].data, topk=(1, 3))
                    top1.update(prec1.item(), input[0].size(0))
                    top3.update(prec3.item(), input[0].size(0))
                elif args.pt_loss == 'TSC':
                    prec1, prec3 = accuracy(output.data, index, topk=(1, 3))
                    top1.update(prec1.item(), output.size(0))
                    top3.update(prec3.item(), output.size(0))
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
                    message = "Finetune Eval: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                    print(message)
                    recorder.record_message('a', message)
                else:
                    for name in ('flip_cls', 'temporal_consistency', 'mutual_loss', 'TSC'):
                        if args.pt_loss == name:
                            message = "Self-supervised Val: Top1:{} Top3:{}".format(top1.avg, top3.avg)
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
    train_transforms, test_transforms, eval_transforms = augmentation_config(args)
    train_data_loader, val_data_loader, _, _, _, _ = data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms)
    # == model config==
    model = model_config(args, num_class)
    recorder.record_message('a', '='*100)
    recorder.record_message('a', str(model.module))
    recorder.record_message('a', '='*100)
    # == optim config==
    train_criterion, val_criterion, optimizer = optim_init(args, model)
    # == data augmentation(self-supervised) config==
    tc = TC(args)
    #========queue init for TemporalDis========================
    # if args.eval_indict == 'loss' and args.pt_loss == 'TemporalDis':
    #     print("initialization TemporalDis Queue")
    #     train(args, tc, train_data_loader, model, train_criterion, optimizer, 1,
    #           recorder, MoCo_init=True)
    #     validate(args, tc, val_data_loader, model, val_criterion, recorder, MoCo_init=True)
    #     print("initialization finished")
    # == train and eval==
    for epoch in range(args.start_epoch, args.epochs):
        timer.tic()
        adjust_learning_rate(optimizer, args.lr, epoch, args.lr_steps)
        if args.eval_indict == 'acc':
            train_prec1, train_loss = train(args, tc, train_data_loader, model, train_criterion, optimizer, epoch, recorder)
            # train_prec1, train_loss = random.random() * 100, random.random()
            recorder.record_train(train_loss / 5.0, train_prec1 / 100.0)
        else:
            train_loss = train(args, tc, train_data_loader, model, train_criterion, optimizer, epoch, recorder)
            # train_prec1, train_loss = random.random() * 100, random.random()
            recorder.record_train(train_loss)
        if (epoch + 1) % args.eval_freq == 0:
            if args.eval_indict == 'acc':
                val_prec1, val_loss = validate(args, tc, val_data_loader, model, val_criterion, recorder)
                # val_prec1, val_loss = random.random() * 100, random.random()
                recorder.record_val(val_loss / 5.0, val_prec1 / 100.0)
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)
                checkpoint = {'epoch': epoch + 1, 'arch': "i3d", 'state_dict': model.state_dict(),
                              'best_prec1': best_prec1}
            else:
                val_loss = validate(args, tc, val_data_loader, model, val_criterion, recorder)
                # val_loss = random.random()
                val_prec1, val_loss = random.random() * 100, random.random()
                recorder.record_val(val_loss)
                is_best = val_loss < lowest_val_loss
                lowest_val_loss = min(val_loss, lowest_val_loss)
                if args.pt_loss == 'TemporalDis':
                    checkpoint = {'epoch': epoch + 1, 'arch': "i3d", 'state_dict': model.module.base_model[1].state_dict(),
                                  'lowest_val': lowest_val_loss}
                else:
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

