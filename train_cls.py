from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
import time
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4) # you can make changes
     
    total_step = len(data_loader)
    # Data size (Each step train on a batch of 4 images)
    data_size = total_step*4

    # Define model, Loss, and optimizer
    model = net(args.num_cls).to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.1, momentum=0.9)

    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        my_lr_scheduler.step()
        print('Current learning rate:{}'.format(get_lr(optimizer)))

        t1 = time.time()
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(data_loader):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            labels = torch.max(labels.long(), 1)[1]
            running_corrects += torch.sum(preds == labels.data)
            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net.ckpt'))

        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.item() / data_size
        print('running_corrects:{}\tdata_size:{}\tepoch_acc:{}'.format(running_corrects,data_size,epoch_acc))
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        t2 = time.time()
        print(t2 - t1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)
