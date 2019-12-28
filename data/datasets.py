import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from config.cfg import cfg


class ScutFBPDataset(Dataset):
    """
    SCUT-FBP dataset
    """

    def __init__(self, f_list, f_labels, transform=None):
        self.face_files = f_list
        self.face_score = f_labels.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], 'SCUT-FBP-%d.jpg' % self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score, 'class': round(score) - 1,
                  'filename': 'SCUT-FBP-%d.jpg' % self.face_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class HotOrNotDataset(Dataset):
    def __init__(self, cv_split_index=1, train=True, transform=None):
        df = pd.read_csv(
            os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0],
                         'eccv2010_split%d.csv' % cv_split_index), header=None)

        filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for _ in df.iloc[:, 0].tolist()]
        scores = df.iloc[:, 1].tolist()
        flags = df.iloc[:, 2].tolist()

        train_set = OrderedDict()
        test_set = OrderedDict()

        for i in range(len(flags)):
            if flags[i] == 'train':
                train_set[filenames[i]] = scores[i]
            elif flags[i] == 'test':
                test_set[filenames[i]] = scores[i]

        if train:
            self.face_files = list(train_set.keys())
            self.face_scores = list(train_set.values())
        else:
            self.face_files = list(test_set.keys())
            self.face_scores = list(test_set.values())

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        score = self.face_scores[idx]

        if score < -1:
            cls = 0
        elif -1 <= score < 1:
            cls = 1
        elif score >= 1:
            cls = 2

        sample = {'image': image, 'score': float(score), 'class': cls,
                  'filename': os.path.basename(self.face_files[idx])}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class SCUTFBP5500Dataset(Dataset):
    """
    SCUT-FBP5500 dataset
    """

    def __init__(self, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(os.path.join(cfg['scutfbp5500_base'],
                                                     'train_test_files/split_of_60%training and 40%testing/train.txt'),
                                        sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['scutfbp5500_base'],
                                                       'train_test_files/split_of_60%training and 40%testing/train.txt'),
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(os.path.join(cfg['scutfbp5500_base'],
                                                     'train_test_files/split_of_60%training and 40%testing/test.txt'),
                                        sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['scutfbp5500_base'],
                                                       'train_test_files/split_of_60%training and 40%testing/test.txt'),
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_base'], 'Images', self.face_img[idx]))
        score = self.face_score[idx]
        sample = {'image': image, 'score': score, 'class': round(score) - 1, 'filename': self.face_img[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class SCUTFBP5500DatasetCV(Dataset):
    """
    Face Dataset for SCUT-FBP5500 with 5-Fold CV
    """

    def __init__(self, cv_index=1, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(
                os.path.join(cfg['scutfbp5500_base'], 'train_test_files', '5_folders_cross_validations_files',
                             'cross_validation_%d' % cv_index, 'train_%d.txt' % cv_index),
                sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(
                os.path.join(cfg['scutfbp5500_base'], 'train_test_files', '5_folders_cross_validations_files',
                             'cross_validation_%d' % cv_index, 'train_%d.txt' % cv_index),
                sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(
                os.path.join(
                    os.path.join(cfg['scutfbp5500_base'], 'train_test_files', '5_folders_cross_validations_files',
                                 'cross_validation_%d' % cv_index), 'test_%d.txt' % cv_index),
                sep=' ',
                header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['scutfbp5500_base'], 'train_test_files',
                                                       '5_folders_cross_validations_files',
                                                       'cross_validation_%d' % cv_index, 'test_%d.txt' % cv_index),
                                          sep=' ',
                                          header=None).iloc[:, 1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_base'], 'Images', self.face_img[idx]))
        attractiveness = self.face_score[idx]
        gender = 1 if self.face_img[idx].split('.')[0][0] == 'm' else 0
        race = 1 if self.face_img[idx].split('.')[0][2] == 'w' else 0

        sample = {'image': image, 'score': attractiveness, 'gender': gender, 'race': race,
                  'class': round(attractiveness) - 1,
                  'filename': self.face_img[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
