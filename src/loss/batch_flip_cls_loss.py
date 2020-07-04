import torch
from torch import nn
from loss.NCECriterion import NCECriterion
from loss.NCEAverage import NCEAverage


class BatchFlipLoss(nn.Module):

    def __init__(self, nLem=3570, flip_num=8, nce=0):
        super(BatchFlipLoss, self).__init__()
        self.nce_criterion = NCECriterion(nLem=nLem)
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()
        self.flip_num = flip_num
        # self.lemniscate = NCEAverage(128, nLem*flip_num, 2048, 0.1, 0.5).cuda()
        self.nce = nce
        if nce:
            self.lemniscate = NCEAverage(128, nLem, 2048, 0.1, 0.5).cuda()

    def batch_flip_loss(self, features):
        # loss = None
        loss = self.nce_criterion(features)
        # for i in range(features.size(0)//self.flip_num):
        #     if loss is None:
        #         loss = self.nce_criterion(features[i::self.flip_num])
        #     else:
        #         loss += self.nce_criterion(features[i::self.flip_num])
        return loss

    def inverse_cls_loss(self, input):
        x, label = input
        return self.cls_criterion(x, label)

    def forward(self, x, indexs):
        # ===============================v1, NCE Loss + CLS Loss====================
        predicts, labels, features = x
        cls_loss = self.inverse_cls_loss((predicts, labels))
        if self.nce:
            feature_invariance_instance = 1.0 / self.flip_num * features[0::self.flip_num, :]
            for i in range(7):
                feature_invariance_instance += 1.0 / self.flip_num * features[i+1::self.flip_num]
            features = self.lemniscate(feature_invariance_instance, indexs.cuda())
            # flip_indexs = torch.ones(predicts.size(0)).long().cuda()
            # for j in range(predicts.size(0)):
            #     flip_indexs[j] = indexs[j // self.flip_num] * self.flip_num + j % 8
            # features = self.lemniscate(features, flip_indexs)
            nce_loss = self.batch_flip_loss(features)
            return 1e-1 * nce_loss + cls_loss
        return cls_loss


class BatchFlipValLoss(nn.Module):

    def __init__(self):
        super(BatchFlipValLoss, self).__init__()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    def batch_flip_loss(self, features):
        loss = self.nce_criterion(features)
        return loss

    def inverse_cls_loss(self, input):
        x, label = input
        return self.cls_criterion(x, label)

    def forward(self, x, indexs):
        predicts, labels, features = x
        cls_loss = self.inverse_cls_loss((predicts, labels))
        return cls_loss
