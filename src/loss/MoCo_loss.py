import torch.nn as nn
import torch
import math

class MoCoLoss(nn.Module):
    def __init__(self, test=False, queue_size = 1024, num_channels=256, tempeature=0.07):
        """
        queue: dictionary as a queue of K keys (C X K)
        :param loss_type:
        :param batch_size:
        :param num_channels:
        :param tempeature:
        """
        super(MoCoLoss, self).__init__()
        self.K = queue_size #queue nums
        stdv = 1. / math.sqrt(num_channels / 3)
        self.register_buffer('memory', torch.rand(self.K, num_channels).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.K, num_channels))
        self.t = tempeature
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        self.test = test
        self.index = 0

    def contrastive_loss(self, q, k, init=False):
        # print(q.size())
        N, C = q.size()
        k = k.detach() # no gradient to keys
        # self.queue = self.queue.detach()
        queue = self.memory.clone().detach() # notice clone()
        # positive logits: Nx1
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(2)
        # negative logits: NxK
        # print(q.size(), self.queue.size())
        l_neg = torch.mm(q.view(N, C), queue.view(C, self.K))
        # logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg], dim=1)
        # print(logits.size())
        # contrastive loss, eqn.(1)
        labels = torch.zeros(N).type(torch.LongTensor).cuda() # postive are the 0-th
        loss = self.ce(logits/self.t, labels)
        # if not self.test and init:
        #     loss.backward()
        # queue and dequeue
        # print(k.size(), self.queue[:,N:].size())
        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(N).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.K) # 1 fmod 1.5 = 1  2 fmod 1.5 = 0.5
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + N) % self.K
        #  self.queue = torch.cat((self.queue[:,N:], k.transpose(0,1)), dim=1)
        return loss

    def forward(self, x, _, init=False):
        q, k = x
        out = self.contrastive_loss(q, k, init)
        return out