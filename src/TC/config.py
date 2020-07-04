from torch import nn
from TC.basic_augmentation.rotation import *
from TC.basic_augmentation.triplet import *
from TC.basic_augmentation.temporal_augment import *
from TC.basic_augmentation.net_mixup import NETMIXUP
import numpy as np
import torch

class TC(nn.Module):
    def __init__(self, args):
        super(TC, self).__init__()
        self.args = args
        if args.pt_loss == 'flip_cls':
            self.mixup = SpatialMixup(0.3, trace=False)
            self.rotation = sample_rotation_cls
            # self.rotation = four_rotation_cls
            # self.rotation = all_flips
        elif args.pt_loss == 'flip':
            self.mixup = SpatialMixup(0.3)
            self.rotation = sample_rotation
        elif args.pt_loss == 'temporal_consistency':
            self.mixup = SpatialMixup(0.3, trace=False)
            self.temporal_augment = temporal_augment
        elif args.pt_loss == 'net_mixup':
            self.net_mixup = NETMIXUP(1)
        elif args.pt_loss == 'mutual_loss':
            self.rotation = four_rotation_cls
        elif args.pt_loss == 'instance_discriminative':
            self.mixup = SpatialMixup(0.3, trace=False)
        elif args.pt_loss == 'TSC':
            print("train temporal sampling classification")
        self.triplet = TRIPLET(t_radio=0.5, s_radio=0.4)
        self.mixup = SpatialMixup(0.3, trace=False)

    def forward(self, input):
        if self.args.eval_indict == 'acc' or self.args.eval_indict == 'feature_extract':
            output = input.cuda()
            # output = self.mixup.mixup_data(output)
            output = torch.autograd.Variable(output)
            return output
        elif self.args.eval_indict == 'loss':
            if self.args.pt_loss == 'flip':
                anchor_input = input[0].cuda()
                postive_input = input[1].cuda()
                rotation_type = np.random.randint(0, 8, size=postive_input.size(0))
                postive_input = self.rotation(postive_input, rotation_type)
                # print(input - rotation_data)
                postive_input = self.mixup.mixup_data(postive_input)
                # anchor_input = self.mixup.mixup_data(anchor_input)
                # print(input - rotation_data)
                # return [input, mixup_data, self.rotation_type]
                anchor_input = torch.autograd.Variable(anchor_input)
                postive_input = torch.autograd.Variable(postive_input)
                # print(input - rotation_data)
                return [anchor_input, postive_input, rotation_type]
            elif self.args.pt_loss == 'flip_cls':
                import random
                anchor_input = input[0].cuda()
                postive_input = input[1].cuda()
                rotation_data = []
                flip_labels = torch.ones(postive_input.size(0) * 8)
                # 16 in total, each time random select four?
                # stt_indexs = random.sample(range(0,16), 4)
                # flip_labels = torch.ones(postive_input.size(0) * 4)
                for i in range(8):
                    # indexs = torch.ones(postive_input.size(0)) * stt_indexs[i]
                    rotation_data.append(self.rotation(postive_input, torch.ones(postive_input.size(0)) * i))
                    flip_labels[postive_input.size(0) * i:postive_input.size(0) * (i + 1)] = torch.ones(postive_input.size(0)) * i
                    # rotation_data.append(self.mixup.mixup_data(self.rotation(postive_input, indexs)))
                    #flip_labels[postive_input.size(0) * i:postive_input.size(0) * (i+1)] = indexs
                return [torch.cat(rotation_data, dim=0), torch.LongTensor(flip_labels.long())]

            elif self.args.pt_loss == 'temporal_consistency':
                import numpy.random as random
                anchor_input = input[0].cuda()
                postive_input = input[1].cuda()
                anchor_rotation_data = []
                postive_rotation_data = []
                b, c, t, h, w = input[0].size()
                # ==== add noise ==========
                # for j in range(t):
                #     spatial_noise = generate_noise((c, h, w), b).cuda()
                #     postive_input[:, :, j, :, :] = (1 - 0.1) * postive_input[:, :, j, :, :] + 0.1 * spatial_noise[:, :, :, :]
                # for j in range(t):
                #     spatial_noise = generate_noise((c, h, w), b).cuda()
                #     anchor_input[:, :, j, :, :] = (1 - 0.1) * anchor_input[:, :, j, :, :] + 0.1 * spatial_noise[:, :, :, :]

                # ======= flip + rotation + mixup
                index = random.randint(0, 15, size=b)
                anchor_rotation_data.append(
                    self.mixup.mixup_data(self.temporal_augment(anchor_input, index)))
                anchor_flip_labels = index

                # postive_input = self.mixup.mixup_data(postive_input)
                index = random.randint(0, 15, size=b)
                postive_rotation_data.append(
                    self.mixup.mixup_data(self.temporal_augment(postive_input, index)))
                postive_flip_labels = index
                return [torch.cat(anchor_rotation_data, dim=0), torch.cat(postive_rotation_data, dim=0),
                        torch.LongTensor(anchor_flip_labels), torch.LongTensor(postive_flip_labels)]

            elif self.args.pt_loss == 'triplet':
                input = input.cuda()
                anchor, postive, negative = self.triplet.construct(input)
                anchor = torch.autograd.Variable(anchor)
                postive = torch.autograd.Variable(postive)
                negative = torch.autograd.Variable(negative)
                return [anchor, postive, negative]
            elif self.args.pt_loss == 'net_mixup':
                prob = self.net_mixup.gen_prob()
                batch, c, t, h, w = input.size()
                a = input[:batch//2]
                b = input[batch//2:]
                mixed_a_b = self.net_mixup.construct(a, b, prob)
                for j in range(t):
                    spatial_noise = generate_noise((c, h, w), batch//2)
                    a[:, :, j, :, :] = (1 - 0.1) * a[:, :, j, :, :] + 0.1 * spatial_noise[:, :, :, :]
                    spatial_noise = generate_noise((c, h, w), batch//2)
                    b[:, :, j, :, :] = (1 - 0.1) * b[:, :, j, :, :] + 0.1 * spatial_noise[:, :, :, :]
                    spatial_noise = generate_noise((c, h, w), batch//2)
                    mixed_a_b[:, :, j, :, :] = (1 - 0.1) * mixed_a_b[:, :, j, :, :] + 0.1 * spatial_noise[:, :, :, :]
                a = torch.autograd.Variable(a)
                b = torch.autograd.Variable(b)
                mixed_a_b = torch.autograd.Variable(mixed_a_b)
                return [a, b, mixed_a_b, prob]

            elif self.args.pt_loss == 'mutual_loss':
                input_a = input[0].cuda()
                input_b = input[1].cuda()
                # print(input_b.size())
                flip_label = np.random.randint(0, 4, input_a.size(0))
                # print(flip_label)
                input_a = self.rotation(input_a, flip_label)
                input_b = self.rotation(input_b, flip_label)
                return [input_a, input_b, torch.LongTensor(flip_label)]
            elif self.args.pt_loss == 'instance_discriminative':
                input_a = input[0].cuda()
                input_b = input[1].cuda()
                # self.mixup.mixup_data(postive_input)
                return [input_a, input_b]
            elif self.args.pt_loss == 'TSC':
                input = torch.cat(input, dim=0)
                output = input.cuda()
                output = torch.autograd.Variable(output)
                return output
            elif self.args.pt_loss == 'TemporalDis':
                input_a = input[0].cuda()
                input_b = input[1].cuda()
                return [input_a, input_b]
            else:
                Exception("unsupported method!")
        else:
            Exception("unsupported method!")
