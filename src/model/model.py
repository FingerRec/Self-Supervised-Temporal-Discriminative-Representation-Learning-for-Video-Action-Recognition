from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


class Sharpen(nn.Module):
    def __init__(self, tempeature=0.5):
        super(Sharpen, self).__init__()
        self.T = tempeature

    def forward(self, probabilities):
        tempered = torch.pow(probabilities, 1 / self.T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)
        return tempered

class MotionEnhance(nn.Module):
    def __init__(self, beta=1, maxium_radio=0.3):
        super(MotionEnhance, self).__init__()
        self.beta = beta
        self.maxium_radio = maxium_radio

    def forward(self, x):
        b, c, t, h, w = x.size()
        mean = nn.AdaptiveAvgPool3d((1, h, w))(x)
        lam = np.random.beta(self.beta, self.beta) * self.maxium_radio
        out = (x - mean * lam) * (1 / (1 - lam))
        return out

class TCN(nn.Module):
    """
    encode a video clip into 128 dimension features and classify
    two implement ways, reshape and encode adjcent samples into batch dimension
    """
    def __init__(self, base_model, out_size, args):
        super(TCN, self).__init__()
        self.base_model = base_model
        self.args = args
        self.l2norm = Normalize(2)
        if self.args.eval_indict == 'loss':
            if self.args.pt_loss == 'flip':
                self.adaptive_pool = nn.AdaptiveMaxPool3d(1)
            elif self.args.pt_loss == 'flip_cls':
                # main stream
                # self.adaptive_pool = nn.AdaptiveMaxPool3d(1)
                # self.feature_embedding = nn.Linear(args.logits_channel, 128)
                # self.flip_classify = nn.Linear(128, 7)
                # se_channel = 64
                # self.main_stream = nn.Sequential(nn.Conv3d(args.logits_channel, se_channel, 1, 1),
                #                                  nn.BatchNorm3d(se_channel),
                #                                  nn.ReLU(True),
                #                                  Flatten(),
                #                                  nn.Linear(se_channel * 8 * 7 * 7, se_channel),
                #                                  nn.BatchNorm1d(se_channel),
                #                                  nn.ReLU(True)
                #                                  )
                # self.flip_classify = nn.Linear(se_channel, 7)
                self.k = 10
                self.rotate_classes = 8 # 4 or 8 or 16
                self.out_size = out_size
                # discriminative filter bancks
                self.G_stream = nn.Sequential(nn.Conv3d(args.logits_channel, self.rotate_classes, 1, 1, 0),
                                              nn.BatchNorm3d(self.rotate_classes),
                                              nn.ReLU(True),
                                              nn.AdaptiveAvgPool3d(1))
                self.conv = nn.Conv3d(args.logits_channel, self.k*self.rotate_classes, 1, 1, 0)
                self.pool = nn.MaxPool3d(self.out_size, self.out_size)
                self.P_stream = nn.Sequential(nn.Conv3d(self.k * self.rotate_classes, self.rotate_classes, 1, 1, 0),
                                              nn.AdaptiveMaxPool3d(1))
                self.cross_channel_pool = nn.AvgPool1d(kernel_size=self.k, stride=self.k, padding=0)

                self.adaptive_pool = nn.AdaptiveMaxPool3d(1)
                self.feature_embedding = nn.Linear(args.logits_channel, 128)

            elif self.args.pt_loss == 'temporal_consistency':
                self.k = 10
                self.num_class = 16
                self.out_size = out_size
                self.G_stream = nn.Sequential(nn.Conv3d(args.logits_channel, 32, 1, 1, 0),
                                              nn.BatchNorm3d(32, affine=False),
                                              nn.ReLU(True),
                                              Flatten(),
                                              nn.Linear(32 * out_size[0] * out_size[1] * out_size[2], self.num_class))
                self.adaptive_pool = nn.AdaptiveMaxPool3d(1)
                self.feature_embedding = nn.Linear(args.logits_channel, 128)
            elif self.args.eval_indict == 'loss' and self.args.pt_loss == 'triplet':
                self.adaptive_pool = nn.AdaptiveMaxPool3d(1)
                self.feature_embedding = nn.Linear(args.logits_channel, 128)
            elif self.args.pt_loss == 'mutual_loss':
                # may need one hidden layer with one relu function
                self.base_G_stream = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                            Flatten(),
                                            nn.Linear(args.logits_channel, 4))
                                            # nn.Linear(args.logits_channel, 128)
                                            # F.relu()
                self.G_stream = nn.Sequential(self.base_G_stream, self.base_G_stream)
            elif self.args.pt_loss == 'instance_discriminative':
                self.hidden_dim = 512
                self.represent_dim = 256
                self.projection = nn.Sequential(nn.AvgPool3d((1,7,7)),
                                                Flatten(),
                                                nn.Linear(args.logits_channel*2, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.represent_dim))
                # self.projection = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                #                             Flatten(),
                #                             nn.Linear(args.logits_channel, self.hidden_dim),
                #                             nn.ReLU(),
                #                             nn.Linear(self.hidden_dim, self.represent_dim))
            elif self.args.pt_loss == 'TSC':
                self.base_stream = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                                   Flatten(),
                                                   nn.Linear(args.logits_channel, 8))
            elif self.args.pt_loss == 'TemporalDis':
                self.projection = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                                Flatten())
                # self.hidden_dim = 512
                # self.represent_dim = 256
                # self.projection = nn.Sequential(nn.AvgPool3d((1, 7, 7)),
                #                                 Flatten(),
                #                                 nn.Linear(args.logits_channel * 2, self.hidden_dim),
                #                                 nn.ReLU(),
                #                                 nn.Linear(self.hidden_dim, self.represent_dim))
            else:
                Exception("not supported pt loss")
        else:
            print("fine tune ...")

    def forward(self, input):
        if self.args.eval_indict == 'acc':
            output = self.base_model(input, return_conv=False)
            # print(output.size())
            # output = F.log_softmax(output, dim=1)
            return output
        elif self.args.eval_indict == 'feature_extract':
            output = self.base_model(input, return_conv=True)
            return output
        elif self.args.eval_indict == 'loss':
            if self.args.pt_loss == 'flip':
                features = torch.cat((input[0], input[1]), 0)
                features = self.base_model(features, return_conv=True)
                # features = self.l2norm(features)
                l_feature = features[:features.size(0)//2]
                lr_feature = features[features.size(0)//2:]
                return l_feature, lr_feature, input[2]
            elif self.args.pt_loss == 'flip_cls':
                # print(input[0].size())
                features = self.base_model(input[0], return_conv=True)
                # print(predict.size())
                # predict = self.main_stream(predict)
                # predict = self.flip_classify(predict)
                # predict = self.flip_classify(predict.squeeze(2).squeeze(2).squeeze(2))
                # predict = self.adaptive_pool(predict).squeeze(2).squeeze(2).squeeze(2)
                # predict = self.feature_embedding(predict)
                # predict = self.flip_classify(predict)
                # G-Stream
                cls_features = self.adaptive_pool(features).squeeze(2).squeeze(2).squeeze(2)
                cls_features = self.feature_embedding(cls_features)
                cls_features = self.l2norm(cls_features)
                x_g = self.G_stream(features)
                x_g = x_g.view(features.size(0), -1)
                # #P-Stream
                x_p = self.conv(features)
                x_p_pool = self.pool(x_p)
                x_p = self.P_stream(x_p_pool)
                x_p = x_p.view(features.size(0), -1)
                # Side-branch
                side = x_p_pool.view(features.size(0), -1, self.k * self.rotate_classes)
                side = self.cross_channel_pool(side)
                side = side.view(features.size(0), -1)
                predict = x_g / 3 + x_p / 3 + side / 3
                # print(out.size())
                _, indices = torch.softmax(predict, dim=1).max(1)
                # # print(torch.softmax(predict, dim=1))
                # print("predict is : {}, real is : {}".format(indices, input[1]))
                return predict, input[1], cls_features
            elif self.args.pt_loss == 'temporal_consistency':
                anchor_features = self.base_model(input[0], return_conv=True)
                anchor_cls_features = self.adaptive_pool(anchor_features).squeeze(2).squeeze(2).squeeze(2)
                anchor_cls_features = self.feature_embedding(anchor_cls_features)
                # anchor_cls_features = self.l2norm(anchor_cls_features)
                anchor_x_g = self.G_stream(anchor_features)
                anchor_x_g = anchor_x_g.view(anchor_features.size(0), -1)
                # ===========postive===============
                postive_features = self.base_model(input[1], return_conv=True)
                postive_cls_features = self.adaptive_pool(postive_features).squeeze(2).squeeze(2).squeeze(2)
                postive_cls_features = self.feature_embedding(postive_cls_features)
                # postive_cls_features = self.l2norm(postive_cls_features)
                postive_x_g = self.G_stream(postive_features)
                postive_x_g = postive_x_g.view(postive_features.size(0), -1)
                return anchor_x_g, postive_x_g, anchor_cls_features, postive_cls_features, input[2], input[3]

            elif self.args.pt_loss == 'triplet':
                features = torch.cat((input[0], input[1], input[2]), 0)
                features = self.base_model(features,  return_conv=True)
                features = self.l2norm(features)
                features = self.adaptive_pool(features).squeeze(2).squeeze(2).squeeze(2)
                features = self.feature_embedding(features)
                anchor = features[:features.size(0)//3]
                postive = features[features.size(0)//3:features.size(0)//3*2]
                negative = features[features.size(0)//3*2:]
                return anchor, postive, negative

            elif self.args.pt_loss == 'net_mixup':
                inputs = torch.cat((input[0], input[1], input[2]), 0)
                features = self.base_model(inputs, return_conv=True)
                a = features[:features.size(0)//3]
                b = features[features.size(0)//3:features.size(0)//3*2]
                mixup_a_b = features[features.size(0)//3*2:]
                return a, b, mixup_a_b, input[3]

            elif self.args.pt_loss == 'mutual_loss':
                for j in range(self.args.mutual_num - 1):
                    # print(self.G_stream[j])
                    path_a = self.base_model[j](input[0], return_conv=True)
                    path_b = self.base_model[j+1](input[1], return_conv=True)
                    # print(self.base_model[j](input[0], return_conv=True) - self.base_model[j+1](input[0], return_conv=True))
                    cls_a = self.G_stream[j](path_a)
                    cls_b = self.G_stream[j+1](path_b)
                return F.log_softmax(cls_a, dim=1), \
                       F.log_softmax(cls_b, dim=1), input[2]

            elif self.args.pt_loss == 'instance_discriminative':
                inputs = torch.cat((input[0], input[1]), 0)
                features = self.base_model(inputs, return_conv=True)
                representation = self.projection(features)
                representation = self.l2norm(representation)
                return representation

            elif self.args.pt_loss == 'TSC':
                features = self.base_model(input, return_conv=True)
                representation = self.base_stream(features)
                return representation

            elif self.args.pt_loss == 'TemporalDis':
                q = self.base_model[0](input[0], return_conv=True)
                k = self.base_model[1](input[1], return_conv=True)
                q = self.projection(q)
                k = self.projection(k)
                q = self.l2norm(q)
                k = self.l2norm(k)
                return q, k
            else:
                Exception("unsupported method")
        else:
            Exception("unsupported method")
