import torch
import torch.nn as nn
from TC.basic_augmentation.rotation import sample_rotation


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


def inverse_loss(input, _):
    """
    question: how to prevent the parameter becomes zeros
    :param x:
    :param x_inverse:
    :return:
    """
    x, x_inverse, rotation_type = input
    # ============================ 3 loss =============================================
    criteion = torch.nn.MSELoss().cuda()
    rotation_data = sample_rotation(x, rotation_type, trace='true')
    # rotation_data = x
    rotation_loss = criteion(x_inverse, rotation_data)
    rotation_data_2 = nn.AdaptiveMaxPool3d(1)(rotation_data).squeeze(2).squeeze(2).squeeze(2)
    x_inverse_2 = nn.AdaptiveMaxPool3d(1)(x_inverse).squeeze(2).squeeze(2).squeeze(2)
    channel_loss = criteion(rotation_data_2, x_inverse_2)
    # ==================== compress all channels into one feature map ==============
    b, c, t, h, w = x_inverse.size()
    x_inverse_3 = x_inverse.view(b, c, -1)
    x_inverse_3 = x_inverse_3.permute(0, 2, 1)
    x_inverse_3 = nn.AdaptiveAvgPool1d(1)(x_inverse_3).squeeze(2)
    rotation_data_3 = rotation_data.view(b, c, -1)
    rotation_data_3 = rotation_data_3.permute(0, 2, 1)
    rotation_data_3 = nn.AdaptiveAvgPool1d(1)(rotation_data_3).squeeze(2)
    map_loss = criteion(x_inverse_3, rotation_data_3)
    # print(rotation_loss.data, channel_loss.data, 10 * map_loss.data)
    loss = rotation_loss + channel_loss + 10 * map_loss
    # loss = channel_loss
    # norm = Normalize(2)
    # x = nn.AdaptiveMaxPool3d(1)(x)
    # x_inverse = nn.AdaptiveMaxPool3d(1)(x_inverse)
    # x = x.view(x.size(0), -1)
    # x_inverse = x_inverse.view(x_inverse.size(0), -1)
    # x = norm(x)
    # x_inverse = norm(x_inverse)
    # criteion = BatchCriterion(1, 0.1, x.size(0))
    # y = torch.cat((x, x_inverse), 0)
    # loss += criteion(y)
    return loss
    """
    norm = Normalize(2)
    x = sample_rotation(x, rotation_type, trace='true')
    # print(x - x_inverse)
    x = nn.AdaptiveMaxPool3d(1)(x)
    x_inverse = nn.AdaptiveMaxPool3d(1)(x_inverse)
    x = x.view(x.size(0), -1)
    x_inverse = x_inverse.view(x_inverse.size(0), -1)
    x = norm(x)
    x_inverse = norm(x_inverse)
    criteion = BatchCriterion(1, 0.1, x.size(0))
    y = torch.cat((x, x_inverse), 0)
    return criteion(y)
    """
    # criteion = torch.nn.MSELoss().cuda()
    # print(x[0, 0, :, :, :])
    # print(x_inverse[0, 0, :, :, :])
    # criteion = torch.nn.L1Loss().cuda()
    # criteion = torch.nn.SmoothL1Loss().cuda()
   #  print(criteion(x, x_inverse))
    # return criteion(x, x_inverse)


def list_inverse_loss(input, _):
    x, x_inverse, rotation_type = input
    loss = None
    for i in range(len(x)):
        if loss is None:
            loss = inverse_loss(x[i], x_inverse[i], rotation_type)
        else:
            loss += inverse_loss(x[i], x_inverse[i], rotation_type)
    return loss


def inverse_cls_loss(input, _):
    criteion = torch.nn.CrossEntropyLoss().cuda()
    x, label = input
    # print(x)
    # print(label)
    return criteion(x, label)
