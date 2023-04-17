import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import albumentations as A
from torch.utils.collect_env import get_pretty_env_info
from torch import nn
torch.backends.cudnn.benchmark = True
from tensorboardX import SummaryWriter
from model4 import *
import scipy.io as scio
import torch.optim as optim

def get_SNR(clean_img, noise_img):
    X_1 = np.sum(np.power(clean_img,2))
    X_2 = np.sum(np.power(noise_img - clean_img,2))
    cal = 10 * np.log10(X_1/X_2)
    return cal



def main():

    
    gpus = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    model_up = UNet()
    pretrained_path1 = '/home/zhangchao/data-disk/fzy/unsupervised/self-supervised-denoise/checkpoint_e1498.pth'
    state1 = torch.load(pretrained_path1)['state_dict']
    new_state_dict1 = {}
    # for k,v in state1.items():
    #     new_state_dict1[k[7:]] = v
    # model_up.load_state_dict(new_state_dict1)
    model_up.load_state_dict(state1)
    model_up = nn.DataParallel(model_up).cuda()

    model_up.eval()

    path = '/home/zhangchao/data-disk/fzy/unsupervised/self-supervised-denoise/moni_noisy.mat'
    full_train_img = scio.loadmat(path)
    full_train_img = np.array(full_train_img['name'])
    data = full_train_img
    data = torch.from_numpy(data)
    data = data.cuda()
    # data = Variable(data)
    data  = torch.unsqueeze(data,0)
    data  = torch.unsqueeze(data,1)
    data = data.type(torch.cuda.FloatTensor)

    mask_path = '/home/zhangchao/data-disk/fzy/unsupervised/pytorch-unsupervised-segmentation-master/moni_4_COLOR_BGR2GRAY_binary.mat'
    mask = scio.loadmat(mask_path)
    mask = np.array(mask['name'])
    mask = torch.from_numpy(mask)
    mask = mask.cuda()
    # data = Variable(data)
    mask  = torch.unsqueeze(mask,0)
    mask  = torch.unsqueeze(mask,1)
    mask = mask.type(torch.cuda.FloatTensor)

    img_path = '/home/zhangchao/data-disk/fzy/unsupervised/self-supervised-denoise/moni.mat'
    clean_img = scio.loadmat(img_path)
    clean_img = np.array(clean_img['seismogram2'])[0:1000,2:170]

    out_seg, out_denoised = model_up(data,mask)
    print(out_seg.shape)
    out_seg = nn.functional.softmax(out_seg,dim = 1)
    out_mask = torch.squeeze(torch.argmax(out_seg,dim = 1)).cpu().numpy()

    
    out_denoised = out_denoised.cpu().detach().numpy().squeeze(0).squeeze(0)
    out_noise = clean_img - out_denoised
    # out_noise = out_noise.cpu().detach().numpy().squeeze(0).squeeze(0)

    out_path = '/home/zhangchao/data-disk/fzy/unsupervised/self-supervised-denoise/denoised_model_moni.mat'
    scio.savemat(out_path, {'name': out_denoised})
    out_path = '/home/zhangchao/data-disk/fzy/unsupervised/self-supervised-denoise/noise_model_moni.mat'
    scio.savemat(out_path, {'name': out_noise})
    
    print(get_SNR(clean_img,out_denoised))

if __name__ == '__main__':
    main()
