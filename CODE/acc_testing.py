#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:33:39 2022

@author: wenting
"""

from tqdm import tqdm
from functionn import *
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()   # set model to evaluation mode
    with torch.no_grad():
        for _step, input_data in enumerate(loader):
            image, label = input_data[0].to(myDevice), input_data[1].to(myDevice) 
            predict_label = myModel.forward(image).to(myDevice)
            a = predict_label.detach().numpy()
            _,preds =  predict_label.max(1) 
            b = preds.detach().numpy()
            num_correct += (preds==label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 *acc ))
        return acc

# load data
train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True, transform=myTransforms)

input_data_size = 1000
noise_rate = 0
train_batch_size = 32

data_value = train_dataset.data
target = train_dataset.targets

#Sample data
data_sample = data_value[:input_data_size, :, :, :]
target_sample = target[:input_data_size]
num_data =  data_sample.shape[0]
classes = list(set(target_sample))

#Model initialization
#Load pretrained model
myModel = torch.load("BD_Noise/BD_clean_2500_20.pkl", map_location=torch.device('cpu'))
myDevice = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

#loading input clean label
# with open("y_train_ns_6.pkl", 'rb') as fb:
#     y_train_ns = pickle.load(fb)

train_dataset.targets = target_sample 
train_dataset.data= data_sample 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle=False, num_workers=0, drop_last=True)
acc = check_accuracy(train_loader, myModel, myDevice)