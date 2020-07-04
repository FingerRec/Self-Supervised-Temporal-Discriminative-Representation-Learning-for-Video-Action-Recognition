import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.05):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, size_average=True):
        anchor, positive, negative = input
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(distance_positive.size()).fill_(1)
        target = target.cuda()
        target = Variable(target)
        loss_triplet = self.criterion(distance_negative, distance_positive, target)
        loss_embedd = anchor.norm(2) + positive.norm(2) + negative.norm(2)
        # print(loss_triplet, 3e-5 * loss_embedd)
        loss = loss_triplet  # + 3e-5 * loss_embedd
        return loss
