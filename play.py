
d = {}

txt = r'D:\DeepLearning\memae-master\memae-master\datasets\Test\Test001\label.txt'
        # 打开存储图像名与标签的txt文件
# fp = open(txt, 'r')
#         # 将图像名和图像标签对应存储起来
# for idx , line in enumerate(fp):
#     line.strip('\n')  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
#     line.rstrip('\n')  # 用来去除结尾字符、空白符(包括\n、\r、\t、’ '，即：换行、回车、制表符、空格)
#     # #string.rstrip([chars]) 参数chars是可选的，当chars为空，默认删除string头尾的空白符(包括\n、\r、\t、’ ')
#     line.rstrip()
#     information = line.split(' ')  # 默认以‘_’ 分割
#     information[1] = information[1].rstrip('\n')
#     d[idx] = information[1]
import torch.nn as  nn

import numpy as np

import torch

def crop_image(img, s):
    # s: cropping size
    if(s>0):
        if(len(img.shape)==3):
            # F(or C) x H x W
            return img[:, s:(-s), s:(-s)]
        elif(len(img.shape)==4):
            # F x C x H x W
            return img[:, :, s:(-s), s:(-s)]
        elif(len(img.shape)==5):
            # N x F x C x H x W
            return img[:, :, :, s:(-s), s:(-s)]
    else:
        return img
recon_frames = torch.randn((1 , 1 ,  16 , 256 , 256))
frames = torch.randn((1 , 1 ,  16 , 256 , 256))
r = recon_frames - frames
# print(r)
sp_error = torch.sum(r**2, dim=1)**0.5
# print(sp_error_map)
print(sp_error.shape)
s = sp_error.size()
sp_error_vec = sp_error.view(frames.shape[0] , 16, -1)
print(sp_error_vec.shape)
recon_error = torch.mean(sp_error_vec, dim=-1)
print(recon_error.shape)
recon_error = (recon_error - recon_error.min())/(recon_error.max() - recon_error.min())
lables = torch.ones(16)
print(recon_error)
print(lables)
lables[3] = 0
from sklearn.metrics import roc_auc_score

print(recon_error[0])
AUC = roc_auc_score(np.array(lables) , np.array(recon_error[0]))

print(AUC)

c = [1.21121 , 2]
print("%lf"%sum(c))


