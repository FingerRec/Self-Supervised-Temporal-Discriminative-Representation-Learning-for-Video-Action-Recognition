import cv2
import torch
import numpy as np

def triplet_visualize(anchor_imgs, positive_imgs, negative_imgs, dir_path):
    #reselect imgs
    anchor = positive_imgs.transpose([ 0, 2, 3, 4, 1])
    positive = anchor_imgs.transpose([0, 2, 3, 4, 1])
    negative = negative_imgs.transpose([0, 2, 3, 4, 1])
    anchor_sp = anchor[0, 0, :, :, :]
    positive_sp = positive[0, 0, :, :, :]
    negative_sp = negative[0, 0, :, :, :]
    # print(anchor_sp)
    mask_img = np.concatenate((anchor_sp, positive_sp, negative_sp), axis=0)
    for i in range(1, anchor.shape[1]):
        if i % 1 == 0:
            save_img(np.uint8(255 * anchor[0, i, :, :, :]), dir_path + '/anchor_' + str(i) + '.png')
            save_img(np.uint8(255 * positive[0, i, :, :, :]), dir_path + '/positive_' + str(i) + '.png')
            save_img(np.uint8(255 * negative[0, i, :, :, :]), dir_path + '/negative_' + str(i) + '.png')
            temp_img = np.concatenate((anchor[0, i, :, :, :], positive[0, i, :, :, :], negative[0, i, :, :, :]), axis=0)
            mask_img = np.concatenate((mask_img, temp_img), axis=1)
    mask_img = np.uint8(255 * mask_img)
    #mask_img = cv2.normalize(mask_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                         dtype=cv2.CV_8UC1)
    # print(mask_img.shape)
    return mask_img

def save_img(img, file_path):
    cv2.imwrite(file_path, img)
    return True