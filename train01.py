
import torch
import torch.nn as nn
from dataset_ import SequenceDataset
from torch.utils.data import DataLoader
from memae_3dconv import AutoEncoderCov3DMem
from  entropy_loss import EntropyLossEncap
import numpy as np
from tqdm import tqdm
class args():
    '''
     python script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P2_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 10 \
    --Seed 1 \
    --SaveCheckInterval 1 \
    --IsTbLog True \
    --IsDeter True \
    --DataRoot ./datasets/processed/ \
    --ModelRoot ./results/ \
    --Suffix Non
    '''
    MemDim = 2000
    EntropyLossWeight=0.0002
    ShrinkThres=0.0025
    # training args
    epochs = 1  # "number of training epochs, default is 2"
    save_per_epoch = 2
    batch_size = 8  # "batch size for training/testing, default is 4"
    pretrained = False
    num_workers = 0
    LR = 1e-4

    resume = False
    saving_model_path = r'memae-master\results'

    # Dataset setting
    channels = 1
    # channels = 1 表示读取灰度，=3 读取RGB
    size = 256
    videos_dir = r'memae-master\datasets\Train'
    time_steps = 16

    # For GPU training
    gpu = 0  # None

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

device = torch.device("cuda")
torch.cuda.set_device(args.gpu)
a  = np.random.choice(100, size=5)
model = AutoEncoderCov3DMem(args.channels,args.MemDim, shrink_thres=args.ShrinkThres).to(device)
recon_loss_func = nn.MSELoss().to(device)
entropy_loss_func = EntropyLossEncap().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
saving_model_path = args.saving_model_path
def train():


    trainloader = DataLoader(dataset=SequenceDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir,
                                                    time_steps=args.time_steps), batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers)
    for epoch_idx in range(0, args.epochs):
        for i, clips in enumerate(trainloader):
            # clips 是一个训练集，其中每5张作为一个训练输入
            pbar = tqdm(clips)
            for j, frames in enumerate(pbar):
                frames = frames.cuda()
                frames = frames.unsqueeze(1)
                print(frames.shape)
                recon_res = model(frames)
                recon_frames = recon_res['output']
                att_w = recon_res['att']
                print(recon_frames.shape)
                loss = recon_loss_func(recon_frames, frames)
                recon_loss_val = loss.item()
                entropy_loss = entropy_loss_func(att_w)
                entropy_loss_val = entropy_loss.item()
                loss = loss + args.EntropyLossWeight * entropy_loss
                loss_val = loss.item()
                ##
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # return
    import datetime
    time_ = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print(time_)
    torch.save(model.state_dict(), saving_model_path)
train()
