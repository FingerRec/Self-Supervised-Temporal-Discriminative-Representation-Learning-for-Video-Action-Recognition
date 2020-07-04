import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

"""
from https://github.com/mangye16/Unsupervised_Embedding_Learning/blob/master/demo_seen.py
in fact, we want to make the features no sensitive to rotation

usage: 
      # define loss function: inner product loss within each mini-batch
        criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size)
        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        inputs = torch.cat((inputs1,inputs2), 0)
        features = net(inputs)
        loss = criterion(features, indexes)
        
"""


class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize * 2).cuda()
        self.batchSize = batchSize

    def tensor_padding(self):
        return None

    def forward(self, x, index=None):
        batchSize = x.size(0)
        self.diag_mat = 1 - torch.eye(batchSize).cuda()
        # if batchSize < self.batchSize:
        # get positive innerproduct
        # narrow (dimension, start, length)
        reordered_x = torch.cat((x.narrow(0, batchSize // 2, batchSize // 2), \
                                 x.narrow(0, 0, batchSize // 2)), 0)
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat
        if self.negM == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum) / batchSize
        return loss


class BatchFlipLoss(nn.Module):

    def __init__(self, nce=0, flip_classes=8, batch_size=1):
        super(BatchFlipLoss, self).__init__()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()
        self.nce = nce
        self.flip_classes = flip_classes
        self.batch_size = batch_size
        if nce:
            self.nce_criterion = BatchCriterion(1, 0.1, batchSize=batch_size)

    def batch_flip_loss(self, features):
        """
        reshape features into shape(bx2,-1), after trasnform, calculate loss
        :param features: b x flip_nums(8)
        :return:
        """
        nce_loss = None
        for i in range(self.flip_classes):
            for j in range(i, self.flip_classes):
                feature_1 = features[i::self.flip_classes, :]
                feature_2 = features[j::self.flip_classes, :]
                inputs = torch.cat((feature_1, feature_2), 0)
                # print(inputs.size())
                part_loss = self.nce_criterion(inputs)
                if nce_loss is None:
                    nce_loss = part_loss
                else:
                    nce_loss += part_loss
        return nce_loss

    def inverse_cls_loss(self, input):
        x, label = input
        return self.cls_criterion(x, label)

    def forward(self, x, indexs):
        # ===============================v1, NCE Loss + CLS Loss====================
        predicts, labels, features = x
        cls_loss = self.inverse_cls_loss((predicts, labels))
        alpha = 3e-2
        if self.nce:
            nce_loss = self.batch_flip_loss(features)
            # print(alpha * nce_loss, cls_loss)
            return alpha * nce_loss + cls_loss
        return cls_loss


class BatchFlipValLoss(nn.Module):

    def __init__(self):
        super(BatchFlipValLoss, self).__init__()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    def inverse_cls_loss(self, input):
        x, label = input
        return self.cls_criterion(x, label)

    def forward(self, x, indexs):
        predicts, labels, features = x
        cls_loss = self.inverse_cls_loss((predicts, labels))
        return cls_loss
