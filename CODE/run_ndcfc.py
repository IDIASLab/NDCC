#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2 10:16:40 2022

@author: wenting
"""
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
from Ndcfc import Ndcfc
from cloth1M_trans import CustomDataset

#Define Input Image Data Transform Format
myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#Customer Input Parameters
input_data_size = 10000
noise_rate = 0.4
noise_type = "AS"
cf_num_iteration = 25
num_training_epoch = 10
train_batch_size = 32
test_batch_size = 32
learning_rate = 0.1
# dataset_name = "Clothing"
# pretrain_model = "cloth_pre_60.pkl"
dataset_name = "CIFAR10"
pretrain_model = "pre_train_model/Resnet50_clean.pkl"

#Prepare and downloas dataets
if (dataset_name =="CIFAR10"):
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=True, transform=myTransforms)
    test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True, transform=myTransforms)
elif (dataset_name == "Clothing"):
    train_path = "clothing_check/validation"
    test_path = "clothing/test"
    train_dataset =  CustomDataset(train_path, transform=myTransforms)		
    test_dataset = CustomDataset(test_path, transform=myTransforms)	
else:
    raise ValueError('Please enter valid dataset name: CIFAR10 or Clothing')


#Prepare pretrain model
myModel = torch.load(pretrain_model, map_location=torch.device('cpu'))
myDevice = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

#Initialize ndcfc class
ns_ndcfc = Ndcfc(input_data_size, train_batch_size, myModel, myDevice, train_dataset, learning_rate)

#Assign noise
noise_index = ns_ndcfc.assign_noise(noise_type, noise_rate)

#Training preperation
data_sample = ns_ndcfc.train_ns_prepare(num_training_epoch, dataset_name)


cf_correct, cf_wrong = ns_ndcfc.training(1)

# for i in range(input_data_size):
#     data, label = train_dataset[i]
#     data_input.append(data)
#     target.append(label)

# data_value_test = train_dataset.data
# target_test = train_dataset.targets
# data_sample_test = data_value_test[:input_data_size, :, :, :]
# target_sample_test = target_test[:input_data_size]


# #Sample data
# num_batch_iter = int(input_data_size/train_batch_size)
# num_data = num_batch_iter * train_batch_size
# classes = list(set(target))
# data_sample =  np.zeros(shape = (num_data, 3,224,224), dtype = float)
# for z in range(num_data):
#     data_sample[z] = data_input[z]

#Load pretrained model


#Adding initial noisy label
# change_label_index_initial, ori_label_initial, ns_label_initial, y_train_ns = assign_nosie(input_data_size*noise_rate, target, num_data)