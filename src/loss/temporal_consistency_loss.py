import torch.nn as nn
import torch


class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize * 2).cuda()

    def forward(self, x):
        batchSize = x.size(0)

        # get positive innerproduct
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


class TemporalConsistencyTrainLoss(nn.Module):

    def __init__(self, nce=0, flip_classes=8, batch_size=1):
        super(TemporalConsistencyTrainLoss, self).__init__()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()
        self.inverse_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.nce = nce
        self.flip_classes = flip_classes
        self.batch_size = batch_size
        if nce:
            self.nce_criterion = BatchCriterion(12, 0.1, batchSize=batch_size)

    def batch_flip_loss(self, features_1, features_2):
        """
        reshape features into shape(bx2,-1), after trasnform, calculate loss
        :param features: b x flip_nums(8)
        :return:
        """
        nce_loss = None
        inputs = torch.cat((features_1, features_2), 0)
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
        anchor_predicts, postive_predicts, anchor_features, postive_features, anchor_labels, postive_labels = x
        # print(anchor_predicts, postive_predicts)
        # print(anchor_labels, postive_labels)
        cls_loss = self.inverse_cls_loss((anchor_predicts, anchor_labels)) \
                   + self.inverse_cls_loss((postive_predicts, postive_labels))
        alpha = 3e-2
        inverse_loss = self.inverse_criterion(anchor_features, postive_features)
        if self.nce:
            nce_loss = self.batch_flip_loss(anchor_features, postive_features)
            print(alpha * nce_loss, cls_loss, inverse_loss)
            return alpha * nce_loss + cls_loss + inverse_loss
        # print(2e-1*cls_loss, 1e1 * inverse_loss)
        return cls_loss + 1e1 *inverse_loss


class TemporalConsistencyValLoss(nn.Module):

    def __init__(self):
        super(TemporalConsistencyValLoss, self).__init__()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    def inverse_cls_loss(self, input):
        x, label = input
        return self.cls_criterion(x, label)

    def forward(self, x, indexs):
        anchor_predicts, postive_predicts, anchor_features, postive_features, anchor_labels, postive_labels = x
        cls_loss = self.inverse_cls_loss((anchor_predicts, anchor_labels)) \
                   + self.inverse_cls_loss((postive_predicts, postive_labels))
        return cls_loss
