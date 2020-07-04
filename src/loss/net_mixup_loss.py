import torch


def mixup_loss(input, _):
    """
    question: how to prevent the parameter becomes zeros
    :param x:
    :param x_inverse:
    :return:
    """
    a, b, mixed_a_b, prob = input
    # ============================ 3 loss =============================================
    criteion = torch.nn.MSELoss().cuda()
    output = prob * a + (1-prob) * b
    loss = criteion(mixed_a_b, output)
    return loss
