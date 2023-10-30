#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

get_ipython().system('pip install torchmetrics')
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import os
import io
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import csv
import random
import matplotlib.pyplot as plt


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# In[9]:


train_number = 50000
val_number = 10000
test_number = 10000
#by set this param you can make your dataset bigger---------for no change in dataset size set it 1
multiple_train_param = 5
multiple_val_param = 5
multiple_test_param = 5


# In[8]:


transform = A.Compose([
                      #  A.HorizontalFlip(p=0.9),
                      #  A.ShiftScaleRotate(p=0.9),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       transforms.ToTensor(),
                      #  A.RandomBrightness(p=0.3),
                       ])

mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor(),)

mnist_train, mnist_val = torch.utils.data.random_split(mnist, [train_number, val_number])

mnist_test = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor(),)


# In[10]:


loaders1 = {
    'train' : torch.utils.data.DataLoader(mnist_train, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1),

    'val' : torch.utils.data.DataLoader(mnist_val, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1),       
    
    'test'  : torch.utils.data.DataLoader(mnist_test, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders1


# In[11]:


for i in range(10):
  plt.imshow(mnist_test.data[i], cmap='gray')
  plt.title(f"{mnist_test.targets[i]}")
  plt.show()


# In[12]:


def dataset_process(train_val_test, writer1 , multiple_dataset_param=1):
  os.makedirs(f"mnist/{train_val_test}/GT", exist_ok=True)
  os.makedirs(f"mnist/{train_val_test}/img", exist_ok=True)

  for batch_idx, (inputs, targets) in enumerate(loaders1[train_val_test]):

    image = inputs[0].permute(1, 2, 0).numpy()
    image_indx = multiple_dataset_param * batch_idx  

    if batch_idx % 10000 == 0:
        print(batch_idx)
#         break
    for index1 in range(multiple_dataset_param):
      angle = random.randint(0, 360)
      transform = A.Compose([
                          A.ShiftScaleRotate(rotate_limit=[angle, angle], border_mode=cv2.BORDER_CONSTANT, p = 1),
                            ])
      transformed = transform(image=image)

      image_loc = f"mnist/{train_val_test}/GT/GT{image_indx+index1}.pt"
      GT_loc = f"mnist/{train_val_test}/img/img{image_indx+index1}.pt"
      writer1.writerow([image_indx+index1, image_loc, GT_loc, targets[0].item(), angle])
      torch.save(np.transpose(image, (2, 0, 1)), GT_loc)
      torch.save(np.transpose(transformed['image'], (2, 0, 1)) , image_loc)


# In[ ]:


# !rm -rf 'mnist'


# In[13]:


def dataset_maker(train_val_test, multiple_train_param):
  f = open(f'./mnist_{train_val_test}.csv', 'w')
  writer = csv.writer(f)
  writer.writerow(['index', 'img_add', 'GT_add','class', 'angle'])
  dataset_process(train_val_test,writer, multiple_train_param)
  f.close()


# In[14]:


dataset_maker('train', multiple_train_param)
dataset_maker('val', multiple_val_param)
dataset_maker('test', multiple_test_param)


# In[19]:


pd.read_csv('mnist_val.csv')


# In[16]:


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_frame, is_google_colab=False):
        'Initialization'
        self.angle = list(data_frame['angle'])
        self.img_add = list(data_frame['img_add'])
        self.GT_add = list(data_frame['GT_add'])
        self.class_ = list(data_frame['class'])
        #if you run this on google colab set this True
        self.is_google_colab = is_google_colab
        
        self.len_ =  len(list(data_frame['index']))



  def __len__(self):
        'Denotes the total number of samples'
        return self.len_


  def __getitem__(self, index):
        # Load data and get lab
        if self.is_google_colab:
          pre = 'content/'
        else:
          pre = './'
        X_add = pre+str(self.img_add[index])
        Y_add = pre+str(self.GT_add[index])
        X = torch.load(X_add)
        Y = torch.load(Y_add)
        extra_params = [self.angle[index], self.class_[index]]

        return X, Y, extra_params


# In[20]:


#if you run this on google colab set this True
is_google_colab = False
mnist_train_info = pd.read_csv('mnist_train.csv')
train_data = Dataset(mnist_train_info, is_google_colab)

mnist_val_info = pd.read_csv('mnist_val.csv')
val_data = Dataset(mnist_val_info, is_google_colab)

mnist_test_info = pd.read_csv('mnist_val.csv')
test_data = Dataset(mnist_test_info, is_google_colab)


# In[21]:


print(len(train_data), len(val_data), len(test_data))


# In[41]:


loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=150, 
                                          shuffle=True, 
                                          num_workers=0),

    'val' : torch.utils.data.DataLoader(val_data, 
                                          batch_size=200, 
                                          shuffle=True, 
                                          num_workers=0),       
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
}
loaders


# In[23]:


inp, res, params = next(iter(loaders['val']))
angles = params[0]
classes = params[1]


# In[24]:


def plot_image(inp, res, angles=None, classes=None, numer=4):
  for i in range(numer):
    plt.imshow(inp[i].permute(1,2,0)[:,:, 0], cmap='gray')
    if angles is not None:
      plt.title(angles[i].item())
    plt.show()
    plt.imshow(res[i].permute(1,2,0)[:,:, 0], cmap='gray')
    plt.show()


# In[25]:


plot_image(inp, res, angles)


# In[26]:


import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('MNIST')


# In[27]:


#sometimes logger.info doesn't work. please make sure this command orint info
logger.info('This is an info message')


# In[29]:


class DoubleConv(nn.Module):
    #(convolution => [BN] => ReLU) * 2

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # Downscaling with maxpool then double conv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    #Upscaling then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# In[30]:


class MNIST_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MNIST_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# In[31]:


net = MNIST_Net(1, 1).to(device=device)


# In[32]:


model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
params


# In[33]:


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)


# In[34]:


criterion(res.to(device), net(inp.to(device)))


# In[35]:


y = torch.tensor( net(inp.to(device)).cpu().detach().numpy() )
plot_image(y, res)


# In[36]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[48]:


# Training
def train(epoch, logger_period=500, val_logger_period=450):
    net.train()
    # train_loss_total = AverageMeter()
    for batch_idx, (inputs, targets, extra_params) in enumerate(loaders['train']):
        inputs, targets, = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # train_loss_total.update(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % logger_period == 0 and batch_idx > 0:
            print(f"Epoch:{epoch} Batch: {batch_idx} Train_Loss: {loss.item():.5}")
#         if batch_idx == 10:
#           break

    net.eval()
    # val_loss_total = AverageMeter()
    for batch_idx, (inputs, targets, extra_params) in enumerate(loaders['val']):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # val_loss_total.update(loss)
        if batch_idx % val_logger_period == 0 and batch_idx > 0 :
            print(f"Epoch:{epoch} Batch: {batch_idx} Val_Loss: {loss.item():.5}")
      
    torch.save(net.state_dict(), 'saved_model.pth')
    torch.save({
    'state_dict': net.state_dict(),
    'optimizer' : optimizer.state_dict(),
    }, 'MNIST_Net.pth.tar')



def test(epoch, checkpoint=0):
    net = MNIST_Net(1, 1).to(device=device)
    checkpoint = torch.load('MNIST_Net.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    net.eval()
    loss_total = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets, extra_params) in enumerate(loaders['test']):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss_total.update(loss)


        print(f'Test:  Epoch:{epoch} Loss:{loss_total.avg:.4}')
        print()
    


# In[39]:


train(0, 50)
test(0)


# In[49]:


criterion(res.to(device), net(inp.to(device)))


# In[50]:


y = torch.tensor( net(inp.to(device)).cpu().detach().numpy() )
plot_image(y, res)


# In[52]:


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.65)


# In[53]:


for i in range(20):
    print(f"Learning Rate For Epoch {i}: {optimizer.param_groups[0]['lr']:.3}")
    train(i, 800, 200)
    scheduler.step()


# In[54]:


criterion(res.to(device), net(inp.to(device)))


# In[55]:


y = torch.tensor( net(inp.to(device)).cpu().detach().numpy() )
plot_image(y, res)


# In[56]:


test(100)


# In[ ]:


def better_pic(input1):
    for k in range(10):
        min1 = np.inf
        max1 = -1 * np.inf
        for i in range(28):
            for j in range(28):
                if input1[k][0][i][j]  < min1:
                    min1 = input1[0][0][i][j]
                if input1[k][0][i][j]  > max1:
                    max1 = input1[0][0][i][j]
        mid1 = (min1 + max1)/2
        print(mid1, min1, max1)
        for i in range(28):
            for j in range(28):
                if input1[k][0][i][j]  < (mid1+min1)/2:
                    input1[k][0][i][j] = 0
    #             elif input1[k][0][i][j]  > (mid1+max1)/1.5:
    #                 input1[k][0][i][j] = 1
    return input1

