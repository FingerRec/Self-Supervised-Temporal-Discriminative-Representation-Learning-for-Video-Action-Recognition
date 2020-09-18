"Self-supervised learning using consistency regularization of spatio-temporal data augmentation for action recognition" [paper](https://arxiv.org/abs/2008.02086)

## Spatio-temporal Transformation

```python
flip_type = flip_type // 4
rot_type = flip_type % 4
# flip at first
for i in range(B):
    if flip_type[i] == 0:
        rotated_data[i] = l_new_data[i]
    elif flip_type[i] == 1:  # left-right flip
        rotated_data[i] = l_new_data[i].flip(3)
    elif flip_type[i] == 2:  # temporal flip
        rotated_data[i] = l_new_data[i].flip(1)
    else:  # left-right + temporal flip
        rotated_data[i] = l_new_data[i].flip(3).flip(1)
# then rotation
for i in range(B):
    if rot_type[i] == 0:
        rotated_data[i] = l_new_data[i]
    elif rot_type[i] == 1:  # 90 degree
        rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)
    elif rot_type[i] == 2:  # 180 degree
        rotated_data[i] = l_new_data[i].flip(2).flip(3)
    else:  # 270 degree
        rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)

```
Please refer to TC/basic_augmentation/rotation for details.

## TCA/[Intra-video Mixup]



The TCA/[Intra-video Mixup] can be implment with fully matrix operation, and can be extend to any video-based self-supervised learning method.

```python
class SpatialMixup(object):
    def __init__(self, alpha, trace=True, version=2):
        self.alpha = alpha
        self.trace = trace
        self.version = version

    def mixup_data(self, x):
        """
        return mixed inputs. pairs of targets
        """
        import random
        if self.version == 1:
        # # ================version 1: random select sample and fusion with stable frame (all video)===================
            b, c, t, h, w = x.size()
            loss_prob = random.random() * self.alpha
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x)
            for i in range(b):
                tmp = (i+1) % b
                img_index = random.randint(t)
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[tmp, :, img_index, :, :]
                    # cv2.imshow("", mixed_x[i,:,j,:,:])
            return mixed_x
        # ================version 2: random select one same video sample and fusion with stable frame=================
        elif self.version == 2:
            b, c, t, h, w = x.size()
            from numpy import random
            loss_prob = random.random() * self.alpha
            # mixed_x = torch.zeros_like(x).cuda()
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x).cuda()
            for i in range(b):
                img_index = random.randint(t)
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[i, :, img_index, :, :]
            return mixed_x
        # #================================ version 3: x and y all change================================
        else:
            b, c, t, h, w = x.size()
            from numpy import random
            loss_prob = random.random() * 0.3
            gama = 3  # control the importance of spatial information
            mixed_x = x
            index = torch.randperm(b)
            for i in range(b):
                img_index = random.randint(t)
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[index[i], :, img_index, :, :]
            # return mixed_x, y, y[index], loss_prob/gama
            return mixed_x

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return (1-lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)

```


```
@Article{wang2020self,
  author  = {Jinpeng Wang and Yiqi Lin and Andy J. Ma},
  title   = {Self-supervised learning using consistency regularization of spatio-temporal data augmentation for action recognition},
  journal = {arXiv preprint arXiv:2008.02086},
  year    = {2020},
}
```