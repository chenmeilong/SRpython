#特征图提取  保存
import os
from tqdm import tqdm          #是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
import h5py
import numpy as np
import argparse                  #是python的一个命令行解析包，用于编写可读性非常好的程序

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import feature_net, classifier

# parse = argparse.ArgumentParser()
# parse.add_argument(
#     '--model', required=True, help='vgg, inceptionv3, resnet152')
# parse.add_argument('--bs', type=int, default=32)
# parse.add_argument('--phase', required=True, help='train, val')
# opt = parse.parse_args()
# print(opt)

img_transform = transforms.Compose([
    transforms.Scale(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = 'kaggle_dog_vs_cat/data'
data_folder = {
    'train': ImageFolder(os.path.join(root, 'train'), transform=img_transform),     #transform数据增强
    'val': ImageFolder(os.path.join(root, 'val'), transform=img_transform)
}

# define dataloader to load images
batch_size =11                ################################resnet152  最多为4      inceptionv3  最多为2    vgg最多为11
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0),
    'val':
    DataLoader(
        data_folder['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
}

# get train data size and validation data size
data_size = {
    'train': len(dataloader['train'].dataset),
    'val': len(dataloader['val'].dataset)
}

# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)

# test if using GPU
use_gpu = torch.cuda.is_available()

def CreateFeature(model, phase, outputPath='.'):
    """
    Create h5py dataset for feature extraction.

    ARGS:
        outputPath    : h5py output path
        model         : used model
        labelList     : list of corresponding groundtruth texts
    """
    featurenet = feature_net(model)
    if use_gpu:
        featurenet.cuda()
    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for data in tqdm(dataloader[phase]):    #tqdm 进度条
        img, label = data               #img   torch.Size([10, 3, 299, 299])        10表示批次
        print (label)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
        out = featurenet(img)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, label), 0)
    feature_map = feature_map.numpy()
    label_map = label_map.numpy()
    file_name = '_feature_{}.hd5f'.format(model)
    h5_path = os.path.join(outputPath, phase) + file_name
    with h5py.File(h5_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)

CreateFeature("vgg", "train")
