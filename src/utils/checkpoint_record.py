import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import datetime
import shutil


class Record:
    def __init__(self, args):
        super(Record, self).__init__()
        self.train_acc_list = list()
        self.val_acc_list = list()
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.path = args.logs_path + '/' + args.method + '_' + args.date
        self.args = args
        if not os.path.exists(args.logs_path):
            os.mkdir(args.logs_path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if self.args.eval_indict == 'acc':
            self.front = 'fine_tune'
        elif self.args.eval_indict == 'loss':
            if self.args.pt_loss == 'flip':
                self.front = 'flip_pt'
            elif self.args.pt_loss == 'triplet':
                self.front = 'triplet_pt'
            else:
                self.front = self.args.pt_loss
        else:
            Exception("wrong")
        self.record_txt = os.path.join(self.path, self.front + '_logs.txt')
        self.record_init(args, 'w')
        self.src_init()
        self.filename = ''
        self.best_name = ''

    def src_init(self):
        if not os.path.exists(self.path + '/src_record'):
            shutil.copytree('../src', self.path + '/src_record')

    def record_init(self, args, open_type):
        with open(self.record_txt, open_type) as f:
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def record_message(self, open_type, message):
        with open(self.record_txt, open_type) as f:
            f.write(message + '\n\n')

    def record_train(self, loss, acc=0):
        self.train_acc_list.append(acc)
        self.train_loss_list.append(loss)

    def record_val(self, loss, acc=0):
        self.val_acc_list.append(acc)
        self.val_loss_list.append(loss)

    def plot_figure(self, plot_list, name='_performance'):
        epoch = len(plot_list[0][0])
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(self.args.arch + '_' + self.front + name)
        for i in range(len(plot_list)):
            plt.plot(axis, plot_list[i][0], label=plot_list[i][1])
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.grid(True)
        plt.savefig(os.path.join(self.path, '{}.pdf'.format(self.front + name)))
        plt.close(fig)

    def save_model(self, model, is_best=False):
        self.save_checkpoint(self.args, model, is_best)
        plot_list = list()
        if self.args.eval_indict == 'acc':
            plot_list.append([self.train_acc_list, 'train_acc'])
            plot_list.append([self.val_acc_list, 'val_acc'])
            plot_list.append([self.train_loss_list, 'train_loss'])
            plot_list.append([self.val_loss_list, 'val_loss'])
        elif self.args.eval_indict == 'loss':
            plot_list.append([self.train_loss_list, 'train_loss'])
            plot_list.append([self.val_loss_list, 'val_loss'])
        else:
            Exception("not supported method!")
        self.plot_figure(plot_list)

    def save_checkpoint(self, args, state, is_best):
        self.filename = self.path + '/' + self.front + '_' + args.mode + '_model_latest.pth.tar'
        torch.save(state, self.filename)
        if is_best:
            self.best_name = self.path + '/' + self.front + '_' + args.mode + '_model_best.pth.tar'
            shutil.copyfile(self.filename, self.best_name)
