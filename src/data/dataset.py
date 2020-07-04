# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random
from data.base import *


class DataSet(data.Dataset):
    def __init__(self, args, root_path, list_file, dataset='ucf101',
                 num_segments=1, new_length=64, stride=1, modality='rgb',
                 image_tmpl='img_{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False, full_video=False):
        self.args = args
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.stride = stride
        self.modality = modality
        self.dataset = dataset
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.full_video = full_video
        if self.args.eval_indict == 'loss':
            self.clips = int(args.clips)
            self.clip_length = new_length // int(self.clips)
        if self.test_mode:
            self.test_frames = 250
        self._parse_list()  # get video list
        # data augmentation


    def _load_image(self, directory, idx):
        directory = self.root_path + directory
        if self.dataset == 'hmdb51':
            directory = "/data1/DataSet/Hmdb51/hmdb51/" + directory.strip().split(' ')[0].split('/')[-1]
        elif self.dataset == 'ucf101':
            directory = "/data1/DataSet/UCF101/jpegs_256/" + directory.strip().split(' ')[0].split('/')[-1]
        else:
            Exception("wrong dataset!")
        if self.modality == 'rgb' or self.modality == 'RGBDiff' or self.modality == 'RGB':
            img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
            # width, height = img.size
            # if width < 256 or height < 256:
            #     # print(width, height)
            #     img = img.resize((max(256, int(256 / height * width)), 256), Image.BILINEAR)
            # if self.args.spatial_size == '112':
            #     width, height = img.size
            #     img = img.resize((int(128 / height * width), 128), Image.BILINEAR)
            return [img]
        elif self.modality == 'flow':
            if self.dataset == 'ucf101':
                u_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
                v_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
                x_img = Image.open(u_img_path).convert('L')
                y_img = Image.open(v_img_path).convert('L')
            else:
                x_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
                y_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record, new_length=32):
        """

        :param record: VideoRecord
        :return: list
        """
        index = random.randint(1, max(record.num_frames - new_length * self.stride, 0) + 1)
        return index  # ? return array,because rangint is 0 -> num-1

    def _get_val_indices(self, record):
        if record.num_frames//2 > self.new_length - 1:
            offsets = np.array(record.num_frames//2)
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record, new_length=64):
        if record.num_frames > self.num_segments + new_length * self.stride - 1:
            offsets = np.sort(
                random.sample(range(0, record.num_frames - new_length * self.stride + 1), self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def get(self, record, indices, new_length=16, is_numpy=False):
        images = list()
        p = int(indices)
        if not self.full_video:
            for i in range(new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
        else:
            p = 1
            if record.num_frames < new_length:
                for i in range(new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
            else:
                for i in range(record.num_frames):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
        # images = transform_data(images, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        if is_numpy:
            frames_up = []
            if self.modality == 'rgb':
                for i, img in enumerate(images):
                    frames_up.append(np.asarray(img))
            elif self.modality == 'flow':
                for i in range(0, len(images), 2):
                    # it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(images[i]), np.asarray(images[i + 1])], axis=2)
                    frames_up.append(tmp)
            images = np.stack(frames_up)

            if self.full_video:
                if record.num_frames < self.new_length:
                    images = self.frames_padding(images, self.new_length)
        return images, record.label

    def get_test(self, record, indices):
        '''
        get num_segments data
        '''
        # print(indices)
        all_images = []
        count = 0
        for seg_ind in indices:
            images = []
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                # print(seg_imgs)
                images.append(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
            all_images.append(images)
            count = count + 1
        process_data = np.asarray(all_images, dtype=np.float32)
        # print(process_data.shape)
        return process_data, record.label

    def get_gl_item(self, index):
        record = self.video_list[index]
        l_segment_indices = random.randint(1, max(2, int((record.num_frames - self.clip_length) / self.clips)))
        l_data = list()
        for i in range(self.clips):
            # loc_index = random.randint(max(1, int(i * record.num_frames / 4 - 10)), i * record.num_frames / 4)
            l_data_temp, label = self.get(record, max(1,int(l_segment_indices + i * record.num_frames / self.clips)
                                                      % record.num_frames), new_length=self.clip_length)
            l_data.append(l_data_temp)
        l_data = np.concatenate(l_data)
        l_data = 2 * (l_data / 255) - 1
        l_data = self.transform(l_data)
        if type(l_data) == list and len(l_data) > 1:
            l_new_data = list()
            for one_sample in l_data:
                l_new_data.append(video_to_tensor(one_sample))
        else:
            l_new_data = video_to_tensor(l_data)
        return l_new_data, label, index

    def get_norm_item(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record, new_length=self.new_length)
            data, label = self.get(record, segment_indices, new_length=self.new_length)
            # data = 2 * (data / 255) - 1
            data = self.transform(data)
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append((one_sample))
            else:
                new_data = (data)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
            data, label = self.get(record, segment_indices, new_length=self.new_length)
            # data = 2 * (data / 255) - 1
            data = self.transform(data)
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append((one_sample))
            else:
                new_data = (data)
        return new_data, label, index

    # pretrain normalizatio
    def get_pre_norm_item(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        # print(segment_indices)
        anchor_data, label = self.get(record, segment_indices)
        # segment_indices = np.array(max(1, segment_indices + random.randint(-4, 4)))
        postive_data, label = self.get(record, segment_indices)
        anchor_data = 2 * (anchor_data / 255) - 1
        anchor_data = self.transform(anchor_data)
        new_anchor_data = video_to_tensor(anchor_data)
        postive_data = 2 * (postive_data / 255) - 1
        postive_data = self.transform(postive_data)
        new_postive_data = video_to_tensor(postive_data)
        return new_anchor_data, new_postive_data, label, index

    # pretrain normalizatio
    def get_moco_items(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        # postive: same in temporal but spatial
        negative_segment_indices = segment_indices
        # important hyperparameter
        thresh = 2
        # count = 0
        # while abs(negative_segment_indices - segment_indices) < thresh:
        #     if not self.test_mode:
        #         negative_segment_indices = self._sample_indices(record)
        #     else:
        #         negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
        #     count += 1
        #     if count > 3:
        #         negative_segment_indices = (segment_indices + random.randint(record.num_frames//8, record.num_frames//3*2)) % record.num_frames
        #         break
        # negative: same in spatial but temporal
        if not self.test_mode:
            negative_segment_indices = self._sample_indices(record)
        else:
            negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
        if abs(negative_segment_indices - segment_indices) < thresh:
            negative_segment_indices = (negative_segment_indices + record.num_frames // 3) % record.num_frames
        if negative_segment_indices == 0:
            negative_segment_indices += 1
        anchor_data, label = self.get(record, segment_indices)
        postive_data, label = self.get(record, segment_indices)
        negative_data, label = self.get(record, negative_segment_indices)
        anchor_data = self.transform(anchor_data)
        postive_data = self.transform(postive_data)
        negative_data = self.transform(negative_data)
        return anchor_data, postive_data, negative_data, label, index

    def __getitem__(self, index):
        if self.args.eval_indict == 'loss':
            if self.args.pt_loss == 'net_mixup':
                data, label, index = self.get_norm_item(index)
                return data, label, index
            elif self.args.pt_loss == 'mutual_loss':
                data, label, index = self.get_norm_item(index)
                origin_data, augment_data, label, index = self.get_pre_norm_item(index)
                return [origin_data, origin_data, data], label, index
            elif self.args.pt_loss == 'TSC':
                datas = []
                strides = []
                for stride in range(1, 9):
                    self.stride = stride
                    data, label, index = self.get_norm_item(index)
                    datas.append(data)
                    strides.append(stride-1)
                return datas, label, torch.tensor(strides)
            # elif self.args.pt_loss == 'MoCo_p':
            #     data, label, index = self.get_norm_item(index)
            #     aug_data, label, index = self.get_norm_item(index)
            #     return [origin_data, augment_data], label, index
            else:
                # #data, label, index = self.get_gl_item(index)
                # origin_data, augment_data, label, index = self.get_pre_norm_item(index)
                # # augment_data = self.mixup.mixup_data(augment_data)
                # # origin_data, label, index = self.get_norm_item(index)
                # # augment_data, _, _ = self.get_norm_item(index)
                # return [origin_data, augment_data], label, index
                anchor_data, postive_data, negative_data, label, index = self.get_moco_items(index)
                return [anchor_data, postive_data, negative_data], label, index
        else:
            data, label, index = self.get_norm_item(index)
            return data, label, index

    def __len__(self):
        return len(self.video_list)
