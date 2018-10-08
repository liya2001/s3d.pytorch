from PIL import Image
import os
import os.path
import torch
import torch.utils.data as data
import numpy as np
from numpy.random import randint
import random

from config import LABEL_MAPPING_2_CLASS, LABEL_MAPPING_3_CLASS, LABEL_MAPPING_2_CLASS2


def costum_collate(batch):
    '''
    Divide frames of one video to multi-segments, as multi batch
    :param batch:
    :return:
    '''
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    # weired not error msg
    if isinstance(batch[0][0], torch.Tensor):
        return batch[0]
    else:
        TypeError((error_msg.format(type(batch[0]))))


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def offset(self):
        return int(self._data[3])

    @property
    def reverse(self):
        return int(self._data[4])


class ViratDataSet(data.Dataset):
    def __init__(self, root_path, list_file, new_length=64, modality='RGB', transform=None,
                 test_mode=False, reverse=False, mapping=None):

        if modality not in ('RGB', 'Flow'):
            raise ValueError('Modality must be RGB or Flow!')

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.test_mode = test_mode
        # For flow
        self.reverse = reverse
        self.mapping = mapping

        self._parse_list()

    def _load_image(self, record, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(self.root_path, record.path, '{}.jpg'.format(idx + record.offset))).convert(
                'RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-x.jpg'.format(idx + record.offset))).convert('L')
            y_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-y.jpg'.format(idx + record.offset))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        Sample load indices
        :param record: VideoRecord
        :return: list
        """
        frame_indices = list(range(record.num_frames))
        rand_end = max(0, len(frame_indices) - self.new_length - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.new_length, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def _get_val_indices_central(self, record):
        frame_indices = list(range(record.num_frames))
        out = frame_indices[max(0, (len(frame_indices) - self.new_length)//2):
                            (len(frame_indices) + self.new_length) // 2]
        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def _get_val_indices_multi_seg(self, record):
        pass

    def __getitem__(self, index):
        record = self.video_list[index]

        # It seems TSN didn't sue validate data set,
        # our val data set is equivalent to TSN model's test set
        if not self.test_mode:
            frame_indices = self._sample_indices(record)
        else:
            frame_indices = self._get_val_indices_central(record)

        return self.get(record, frame_indices)

    def get(self, record, indices):

        images = list()
        for idx in indices:
            images.extend(self._load_image(record, idx))

        # For flow, reverse image for data augmentation
        # reverse_flag = random.random() >= 0.5
        # and record.reverse
        # if self.reverse and reverse_flag:
        #     images = images[::-1]

        process_data = self.transform(images)
        # ToDo: just for 2 classify
        # 1 if record.label > 0 else 0 LABEL_MAPPING_2_CLASS
        if self.mapping:
            label = self.mapping[record.label]
        else:
            label = record.label

        return process_data, label

    def __len__(self):
        return len(self.video_list)


class ViratValDataSet(data.Dataset):
    def __init__(self, root_path, list_file, new_length=64, modality='RGB', transform=None,
                 num_segments=3, test_mode=False, reverse=False, mapping=None):

        if modality not in ('RGB', 'Flow'):
            raise ValueError('Modality must be RGB or Flow!')

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.num_segments = num_segments
        self.test_mode = test_mode
        # For flow
        self.reverse = reverse
        self.mapping = mapping

        self._parse_list()

    def _load_image(self, record, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(self.root_path, record.path, '{}.jpg'.format(idx + record.offset))).convert(
                'RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-x.jpg'.format(idx + record.offset))).convert('L')
            y_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-y.jpg'.format(idx + record.offset))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        Sample load indices
        :param record: VideoRecord
        :return: list
        """
        frame_indices = list(range(record.num_frames))
        rand_end = max(0, len(frame_indices) - self.new_length - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.new_length, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def _get_val_indices_central(self, record):
        frame_indices = list(range(record.num_frames))
        out = frame_indices[max(0, (len(frame_indices) - self.new_length)//2):
                            (len(frame_indices) + self.new_length) // 2]
        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def temperal_transform(self, indices):
        for index in indices:
            if len(indices) >= self.new_length:
                break
            indices.append(index)

        return indices

    def _get_val_indices_multi_seg(self, record):
        frame_indices = list(range(record.num_frames))
        step = max(0, (record.num_frames - self.new_length) // (self.num_segments - 1))

        multi_frame_indices = []
        for i in range(self.num_segments):
            indices = self.temperal_transform(frame_indices[i*step:i*step+self.new_length])
            if len(indices) == 0:
                print(step, i, frame_indices, frame_indices[i*step:i*step+self.new_length])
            multi_frame_indices.append(self.temperal_transform(frame_indices[i*step:i*step+self.new_length]))

        return multi_frame_indices

    def __getitem__(self, index):
        record = self.video_list[index]

        # It seems TSN didn't sue validate data set,
        # our val data set is equivalent to TSN model's test set
        if not self.test_mode:
            # frame_indices = self._sample_indices(record)
            raise ValueError('Only for val/test')
        else:
            multi_frame_indices = self._get_val_indices_multi_seg(record)
        return self.get_multi_seg(record, multi_frame_indices)
        # return self.get(record, frame_indices)

    def get(self, record, indices):
        assert len(indices) > 0
        images = list()
        for idx in indices:
            images.extend(self._load_image(record, idx))

        # For flow, reverse image for data augmentation
        # reverse_flag = random.random() >= 0.5
        # and record.reverse
        # if self.reverse and reverse_flag:
        #     images = images[::-1]

        process_data = self.transform(images)
        # ToDo: just for 2 classify
        # 1 if record.label > 0 else 0 LABEL_MAPPING_2_CLASS
        if self.mapping:
            label = self.mapping[record.label]
        else:
            label = record.label

        return process_data, label

    def get_multi_seg(self, record, multi_indices):
        batch = []
        for frame_indices in multi_indices:
            processed_data, label = self.get(record, frame_indices)
            batch.append(processed_data)

        batch = torch.stack(batch, dim=0)
        label = torch.LongTensor([label])
        return batch, label

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    pass

