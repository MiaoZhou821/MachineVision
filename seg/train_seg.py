from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from torch.utils.data import Dataset, DataLoader
from cfg.cfg_a2d import train as train_cfg
from cfg.cfg_a2d import val as val_cfg
from cfg.cfg_a2d import test as test_cfg
from torch.optim import lr_scheduler
#from network_fcn import FCN
#from network_fcn16 import FCN16
#from fcn32s import FCN32s
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torchfcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_encode(labels, num_cls,device):
    labels_extend = labels.clone()
    labels_extend.unsqueeze_(1)
    one_hot = torch.cuda.LongTensor(labels_extend.size(0), num_cls, labels_extend.size(2), labels_extend.size(3)).zero_().to(device)
    labels_extend = labels_extend.long()
    target = one_hot.scatter(1,labels_extend.type(torch.long),1)
    target = target.type(torch.float32)
    return target

def get_parameters(model,bias=False):
    modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            torchfcn.models.FCN32s
            )
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight

        elif isinstance(m,nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m,nn.ConvTranspose2d):
            if bias:
                assert m.bias is None
        elif isinstance(m,modules_skipped):
            continue
        else:
            raise ValueError('Unexpected modules: %s' %str(m))

def cross_entropy2d(input,target,weight=None,size_average=False):
    n,c,h,w = input.size()
    log_p = F.log_softmax(input,dim=1)

    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    log_p = log_p[target.view(n,h,w,1).repeat(1,1,1,c) >=0]
    log_p = log_p.view(-1,c)

    mask = target>=0
    target = target[mask]
    loss = F.nll_loss(log_p,target,weight=weight,reduction='sum')
    if size_average:
        loss/=mask.data_sum()
    return loss


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg)
    data_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    model = torchfcn.models.FCN32s(n_class=args.num_cls).to(device)
    #vgg16 = torchfcn.models.VGG16(pretrained=True).to(device)
    #model.copy_params_from_vgg16(vgg16)
    model.load_state_dict(torch.load(os.path.join(args.model_path,'net.ckpt')))
    #criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(
            [
                {'params':get_parameters(model,bias=False)},
                {'params':get_parameters(model,bias=True),
                'lr':1e-10*2, 'weight_decay':0},
            ],
            lr=1e-10,
            momentum=0.99,
            weight_decay = 0.00005)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.99)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma =0.1)
    total_step = len(data_loader)
    weight = torch.ones(44).to(device)
    weight[0] = 1e-2
    for epoch in range(args.num_epochs):
        #mask_list = []
        t1 = time.time()
        for i, data in enumerate(data_loader):
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(images)
            loss= cross_entropy2d(outputs,labels)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net_Wm2.ckpt'))
        t2 = time.time()
        print(t2 - t1)
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
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)
