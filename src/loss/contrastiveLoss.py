import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, loss_type=0, batch_size=3, num_channels=128, tempeature=0.07, test=False):
        """
        queue: dictionary as a queue of K keys (C X K)
        :param loss_type:
        :param batch_size:
        :param num_channels:
        :param tempeature:
        """
        super(ContrastiveLoss, self).__init__()
        self.loss_type = loss_type
        self.K = 30 #queue nums
        self.queue = torch.ones(num_channels, self.K).cuda() * 1e-5
        self.t = tempeature
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        self.test = test

    def NTXent(self, p, n):
        # print(p.size())
        sample_num = p.size(0)
        loss = 0
        tempeature = 2
        sim_all = 0
        sim_s = 0
        for i in range(sample_num):
            for j in range(i, sample_num):
                sim_all += torch.exp(F.cosine_similarity(p[i], p[j], dim=0)/tempeature)
                sim_all += torch.exp(F.cosine_similarity(p[i], n[j], dim=0) / tempeature)
                sim_all += torch.exp(F.cosine_similarity(n[i], n[j], dim=0) / tempeature)
                sim_all += torch.exp(F.cosine_similarity(n[i], p[j], dim=0) / tempeature)
        for i in range(sample_num):
            sim_s += torch.exp(F.cosine_similarity(p[i], n[i], dim=0) / tempeature)
        return -torch.log(sim_s/sim_all)

    def NTDot(self, q, k):
        N, C = q.size()
        k = k.detach() # no gradient to keys
        self.queue = self.queue.detach()
        # positive logits: Nx1
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(2)
        # negative logits: NxK
        # print(q.size(), self.queue.size())
        l_neg = torch.mm(q.view(N, C), self.queue.view(C, self.K))
        # logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg], dim=1)
        # print(logits.size())
        # contrastive loss, eqn.(1)
        labels = torch.zeros(N).type(torch.LongTensor).cuda() # postive are the 0-th
        loss = self.ce(logits/self.t, labels)
        if not self.test:
            loss.backward()
        # queue and dequeue
        # print(k.size(), self.queue[:,N:].size())
        self.queue = torch.cat((self.queue[:,N:], k.transpose(0,1)), dim=1)
        return loss

    def MarginTriplet(self, p, n):
        return 3

    def forward(self, x, _):
        anchor = x[:x.size(0) // 2]
        postive = x[x.size(0) // 2:]
        if self.loss_type == 0:
            out = self.NTXent(anchor, postive)
        elif self.loss_type == 1:
            out = self.NTDot(anchor, postive)
        else:
            out = self.MarginTriplet(anchor, postive)
        # print(out)
        return out