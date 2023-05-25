
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2

import numpy as np

import os
import glob

class SequenceDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):

        self.videos = []
        self.seqs_idx = []

        for f in sorted(os.listdir(videos_dir)):
            print(os.path.join(videos_dir, f, '*.tif'))
            frames = glob.glob(os.path.join(videos_dir, f, '*.tif'), recursive=True)

            frames.sort()
            self.videos.append(frames)

            selected_idx = np.random.choice(len(frames) - time_steps, size=5)

            self.seqs_idx.append(selected_idx)

        self.time_steps = time_steps
        self.size = size

        self.channels = channels

    def __len__(self):                    
        return len(self.videos)

    def __getitem__(self, index):

        video = self.videos[index]

        selected_idx = self.seqs_idx[index]

        clips = []
        for idx in selected_idx:
            frames = video[idx:idx + self.time_steps]

            if self.channels == 1:
                frames1 = []
                for frame in frames:
                    frames1.append(cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32))
                frames = frames1
            else: 
                frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in frames]
                    
            frames = [simple_transform(frame, self.size, self.channels) for frame in frames]

            frames = torch.stack(frames)
            frames = torch.reshape(frames, (-1, self.size, self.size))
            clips.append(torch.tensor(frames))

        return clips


class TestDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):
        self.videos_dir = videos_dir
        self.videos = glob.glob(os.path.join(videos_dir, '*.tif'), recursive=True)
        self.videos.sort()

        self.time_steps = time_steps
        self.size = size

        self.channels = channels

        self.selected_idx = np.random.choice(len(self.videos) - time_steps, size=20)


    def __len__(self):                    
        return len(self.videos) - self.time_steps

    def __getitem__(self, index):
        lable = {}
        txt = self.videos_dir + r'\label.txt'
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        # 将图像名和图像标签对应存储起来
        for idx , line in enumerate(fp):
            line.strip('\n')  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            line.rstrip('\n')  # 用来去除结尾字符、空白符(包括\n、\r、\t、’ '，即：换行、回车、制表符、空格)
            # #string.rstrip([chars]) 参数chars是可选的，当chars为空，默认删除string头尾的空白符(包括\n、\r、\t、’ ')
            line.rstrip()
            information = line.split(' ')  # 默认以‘_’ 分割
            information[1] = information[1].rstrip('\n')
            lable[idx] = information[1]

        selected_idx = self.selected_idx

        seqs = []
        lables = []
        for idx in selected_idx:
            frames = self.videos[idx:idx + self.time_steps]
            frames1 = []
            f_lable = []
            if self.channels == 1:
                for id , frame in enumerate(frames):
                    frames1.append(cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32))
                    f_lable.append(int(lable[idx+id]))
                frames = frames1


            else:
                frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in self.videos[index:index + self.time_steps]]
            frames = [simple_transform(frame, self.size, self.channels) for frame in frames]
            frames = torch.stack(frames)
            frames = torch.reshape(frames, (-1, self.size, self.size))
            seqs.append(torch.tensor(frames))
            lables.append(f_lable)

        return seqs, np.array(lables)

def simple_transform(img, size, channels):

    if channels == 1:
        mean = 0.5
        std = 0.5
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = A.Compose([
                            A.Resize(height=size,
                                    width=size,
                                    always_apply=True,
                                    p=1.0),
                            A.Normalize(mean=mean,
                                        std=std,
                                        max_pixel_value=255.0,
                                        p=1.0),
                            ToTensorV2(p=1.0)
                        ], p=1.0)

    img = transform(image=img)['image']

    return img

def base_transform(img, size):
    transform = A.Compose([
                            A.Resize(height=size,
                                    width=size,
                                    always_apply=True,
                                    p=1.0)
        , ToTensorV2(p=1.0)
                        ], p=1.0)

    img = transform(image=img)['image']

    return img

    