import torch.nn as nn
import torch

def mutual_loss(input, _):
    a, b, target = input
    # print(a, target)
    cls_criteation = nn.NLLLoss()
    kl_diverge = nn.KLDivLoss(reduction='sum')
    cls_loss = cls_criteation(a, target) + cls_criteation(b, target)
    kl_loss = kl_diverge(a, torch.exp(b)) + kl_diverge(b, torch.exp(a))
    # print(cls_loss, kl_loss)
    loss = cls_loss + 10 * kl_loss
    return loss