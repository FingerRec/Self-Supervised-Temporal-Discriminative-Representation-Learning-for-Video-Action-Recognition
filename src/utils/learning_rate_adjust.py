import numpy as np


def adjust_learning_rate(optimizer, intial_lr, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.3 ** (sum(epoch >= np.array(lr_steps)))
    lr = intial_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

