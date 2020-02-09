from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader

from cfg.cfg_a2d import train as train_cfg
from cfg.cfg_a2d import val as val_cfg
from cfg.cfg_a2d import test as test_cfg
import pickle
import pdb
import torchfcn
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def oneHot2One(output,device):#output:[classes,w,h]
    #print("output:",output.size())
    _,w,h = output.size()
    result = torch.zeros((w,h)).to(device)
    for i in range(w):
        for v in range(h):
            vertical = output[:,i,v]
            result[i,v] = torch.argmax(vertical)
    #print("result:",result.size())

def predict(args):
    
    #val_dataset = a2d_dataset.A2DDataset(val_cfg)
    test_dataset = a2d_dataset.A2DTestDataset(test_cfg)
    data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)

    
    model = torchfcn.models.FCN32s(args.num_cls).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path,'net.ckpt')))
    
    gt_list = []
    mask_list = []  
    acc = 0
    iu = 0
    model.eval()
    with torch.no_grad():
        for batch_idx,data in enumerate(data_loader):

            print("step:{}/{}".format(batch_idx,len(test_dataset)))
            #images = data[0].to(device)
            #gt = data[1].to(device)#[224,224]
            #print('gt:',np.count_nonzero(gt.cpu().numpy()))
            images = data.to(device)
            output = model(images)
            #mask = oneHot2One(output,device)#[224,224]
            mask = output.data.max(1)[1].cpu().numpy()[:,:,:]
            mask = mask.astype(np.uint8)
            mask_list.append(mask)
            #gt = gt.cpu().numpy()
            #gt = gt.astype(np.uint8)
            #gt_list.append(gt)
        #gt_list = np.array(gt_list)
        #mask_list = np.array(mask_list)
    #with open('mask_FCN_zkf.pkl', 'wb') as f:
        #pickle.dump(mask_list,f)
    #with open('mask_gt_zkf.pkl','wb') as f:
        #pickle.dump(gt_list,f)
    with open('test_zkf.pkl','wb') as f:
        pickle.dump(mask_list,f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=44)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    print(args)
predict(args)
            



