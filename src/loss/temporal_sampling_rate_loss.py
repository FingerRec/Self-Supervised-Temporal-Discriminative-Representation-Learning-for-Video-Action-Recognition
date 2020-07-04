import torch

def tsc_loss(input, _):
    criteion = torch.nn.CrossEntropyLoss().cuda()
    labels = []
    loss = 0
    for i in range(input.size(0)//8):
        for j in range(8):
            labels.append(j)
            loss += criteion(input[i*8+j].view(1,input.size(1)), torch.tensor(j).view(1,1).cuda())
    return loss
