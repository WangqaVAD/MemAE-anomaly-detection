import torch
from torch.utils.data import DataLoader
from dataset_ import TestDataset
from memae_3dconv import AutoEncoderCov3DMem
from  entropy_loss import EntropyLossEncap
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2
import imageio




class args():
    # Model setting
    MemDim = 2000
    EntropyLossWeight = 0.0002
    ShrinkThres = 0.0025
    checkpoint = r'memae-master/results/1.pth'

    # Dataset setting
    channels = 1
    size = 256
    videos_dir = r'memae-master\datasets\Test\Test001'
    time_steps = 16

    # For GPU training
    gpu = 0  # None


device = torch.device("cuda")

def evaluate():
    model = AutoEncoderCov3DMem(args.channels,args.MemDim, shrink_thres=args.ShrinkThres).to(device)
    model_para = torch.load(args.checkpoint)
    model.load_state_dict(model_para)
    model.to(device)
    model.eval()

    testloader = DataLoader(dataset=TestDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir,
                                                time_steps=args.time_steps), batch_size=1, shuffle=False, num_workers=0)


    with torch.no_grad():
        total_AUC = []
        for i, datas in enumerate(tqdm(testloader)):

            seqs, lables = datas[0], datas[1]
            lables = lables[0]
            pbar = tqdm(seqs)
            for j, frames in enumerate(pbar):
                if sum(lables[j]) == 0 or sum(lables[j]) == 16 : continue
                frames = frames.cuda()
                frames = frames.unsqueeze(1)
                print(frames.shape)
                # print(lables[j].shape)
                recon_res = model(frames)
                recon_frames = recon_res['output']
                r = recon_frames - frames
                sp_error = torch.sum(r ** 2, dim=1) ** 0.5
                # print(sp_error_map)
                print(sp_error.shape)
                sp_error_vec = sp_error.view(frames.shape[0], args.time_steps, -1)
                print(sp_error_vec.shape)
                recon_error = torch.mean(sp_error_vec, dim=-1)
                print(recon_error.shape)
                recon_error = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())
                print(recon_error)
                print(lables[j])
                print(recon_error[0])
                AUC = roc_auc_score(np.array(lables[j].cpu()), np.array(recon_error[0].cpu()))
                print(AUC)
                total_AUC.append(AUC)
            if(i == 2):
                print("total Mean AUC : %lf" %(sum(total_AUC)/len(total_AUC)))
                break



evaluate()
