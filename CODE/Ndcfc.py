#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2 10:21:57 2022

@author: wenting
"""
from tqdm import tqdm
from functionn import *
from noisy_assign import assign_sym_nosie, assign_asym_nosie
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
from torchmetrics.image.kid import KernelInceptionDistance

class RESNET_MF(nn.Module):
    def __init__(self, X_ori):
        super(RESNET_MF, self).__init__()
        self.x = torch.nn.Parameter(X_ori, requires_grad=True)
    def forward(self, model):#, t1, b1, t2, b2, t3, b3):
        self.o2 =model(self.x)
        return self.o2, self.x    
    
class Ndcfc():
    def __init__(self, input_data_size, train_batch_size, myModel, myDevice, train_dataset, learning_rate):
        self.train_dataset = train_dataset
        self.size = input_data_size
        self.nr = 0
        self.train_bs = train_batch_size
        self.myModel = myModel
        self.myDevice = myDevice
        self.noisy_index = None
        self.x = []
        self.y = []
        self.y_ori = []
        self.classes = []
        self.lr = learning_rate
        
        self.myOptimzier_ns = optim.AdamW(myModel.parameters(), lr = learning_rate)
        
        for i in range(20000, 20000+self.size):
            data, label = train_dataset[i]
            self.x.append(data)
            self.y.append(label)
            self.y_ori.append(label)
    
    def assign_noise(self, noise_type, noise_rate):
        self.nr += noise_rate
        num_batch_iter = int(self.size/self.train_bs)
        num_data = num_batch_iter * self.train_bs
        if noise_type == "S":
            self.noisy_index, self.ori_label, self.ns_label, self.y = assign_sym_nosie(self.size*self.nr, self.y, num_data)
        elif noise_type == "AS" :
            self.noisy_index, self.ori_label, self.ns_label, self.y = assign_asym_nosie(self.size*self.nr, self.y, num_data)
        else:
            raise ValueError('Please enter valid noisy type: S or AS')
        return self.noisy_index
    
    def train_ns_prepare(self, num_iteration, dataset):
        self.num_batch_iter = int(self.size/self.train_bs)
        num_data = self.num_batch_iter * self.train_bs
        self.classes = list(set(self.y))
        self.data_sample =  np.zeros(shape = (num_data, 3,224,224), dtype = float)
        for z in range(num_data):
           self.data_sample[z] = self.x[z]
         
        self.loss_record_mat_ori = np.zeros(shape = (num_data, num_iteration))
        self.loss_record_mat_all = np.zeros(shape = (num_data, num_iteration))
        self.loss_record_mat_ns = np.zeros(shape = (num_data, num_iteration))
        self.loss_record_mat_pre_label = np.zeros(shape = (num_data, num_iteration))
        
        if(dataset == "Clothing"):
            self.train_dataset.targets = self.y
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.train_bs, shuffle=False, num_workers=0, drop_last= False)
        else:
            self.data_value_test = self.train_dataset.data
            self.data_sample_test = self.data_value_test[20000:20000 + self.size, :, :, :]
            self.train_dataset.targets = self.y
            self.train_dataset.data= self.data_sample_test
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.train_bs, shuffle=False, num_workers=0, drop_last= False)
        return self.y
    
    def training(self, num_iteration):
        for tt in tqdm(range(num_iteration)):
            epoch =  tt
            self.myModel.eval() ###Chaning Place 1
        # Noise detection training process
            myLoss = torch.nn.CrossEntropyLoss()
            training_loss = 0.0
            data_pre_num = 0
            loss_value_clean  = torch.Tensor([0])
            for _step, input_data in enumerate(self.train_loader):
                # loss_value_clean = torch.Tensor([0])
                if (_step < self.num_batch_iter):
                    print(_step)
                    image, label = input_data[0].to(self.myDevice), input_data[1].to(self.myDevice)   # GPU加速
                    data_pre_num  = _step * (len(label))
                    predict_label = self.myModel.forward(image).to(self.myDevice)
                    label_get = list(predict_label.max(1))
                    self.loss_record_mat_pre_label[data_pre_num: data_pre_num +len(label) ,tt] = label_get[1].numpy()
                    # loss_value_temp = myLoss(predict_label, label)
                    # loss_value_clean = loss_value_temp + loss_value_clean
                    for m in range(len(label)):
                        ori_loss = myLoss(predict_label[m,:].reshape([1,-1]), label[m].reshape([1]))
                        self.loss_record_mat_ori[m + data_pre_num, epoch] = ori_loss.detach()
                        thr_inital = torch.Tensor([0])
                        for j in range(len(self.classes)):
                            temp_value = myLoss(predict_label[m,:].reshape([1,-1]), torch.LongTensor([self.classes[j]]).reshape([1]).to(self.myDevice))
                            thr_inital = thr_inital.to(self.myDevice) + temp_value.to(self.myDevice)
                        ave_loss = thr_inital/len(self.classes)
                        all_loss = ori_loss - ave_loss 
                        if (all_loss < 0):
                            self.loss_record_mat_ns[m + data_pre_num, epoch] = 1
                            self.loss_value_clean= loss_value_clean.to(self.myDevice) + ori_loss.to(self.myDevice)
                        self.loss_record_mat_all[m + data_pre_num, epoch] = all_loss.detach()
                else:
                    break
            print("Finish training noise detection model")   
            
            cluster_centroid = {}
            loss_ori_list = list(self.loss_record_mat_ori[:, 0])
            predict_label_list = list(self.loss_record_mat_pre_label[:, 0])
            for m in range(len(loss_ori_list)):
                label_value = self.y[m]
                if (label_value not in cluster_centroid.keys()):
                    cluster_centroid.setdefault(label_value, [self.x[m], loss_ori_list[m]])

                else:
                    if (loss_ori_list[m] < cluster_centroid[label_value][1]):
                        print(m, label_value)
                        print("predict_label: {}".format(predict_label_list[m]))
                        cluster_centroid[label_value] = [self.x[m],  loss_ori_list[m]]
                        
                
            torch.save(self.myModel, 'Resnet50_test.pkl') # 保存整个模型     
            
            #Output detected noise label
            ns_data_index, ns_label, ns_data = ns_detect(self.loss_record_mat_ns, self.y, self.data_sample, epoch)
                
            #Count_true_detect, count_false_detect, wrong_detect, miss_detacted
            true_detect, wrong_detect, miss_detect = noise_detect_result_analysis(ns_data_index, self.noisy_index, self.ori_label, self.ns_label)
            print("The number of true ns detect is {}".format(len(true_detect)))
            print("The number of wrong ns detect is {}".format(len(wrong_detect)))
            print("The number of miss ns detect is {}".format(len(miss_detect)))
            print("Start training noise correction model")
            
            cf_model = torch.load("Resnet50_test.pkl").eval()
            cf_correct_data = [] 
            myLoss_cf = torch.nn.CrossEntropyLoss()
            for z in range(len(ns_data)):
                pos_label_list = self.classes
                cf_info = {}
                data_temp = ns_data[z, :,:, :]
                test_data = data_temp 
                X_unchange = test_data
                for k in range(len(self.classes)):
                    tlnn2  = RESNET_MF((cluster_centroid[k][0] - 0).reshape([1, 3, 224, 224]).to(self.myDevice))
                    # tlnn2  = RESNET_MF(data_cluster_centroid[k].to(myDevice))
                    optimizer_cf = torch.optim.AdamW(tlnn2.parameters(), lr = 0.2)
                    Xpred1, cf_data = tlnn2(cf_model)  
                    _, predict_label_cf = Xpred1.max(1)
                    label_pos_list = [self.classes[k]]
                    dis_loss_list = []
                    valid_loss_list = []
                    cf_data_index = []
                    count_num = 0
                    vali_loss = 10
                    predict_label_cf = k
                    while  (count_num < 50 and predict_label_cf == k) :
                        Xpred1, cf_data = tlnn2(cf_model)  
                        _, predict_label_cf = Xpred1.max(1)
                        dis_loss = torch.norm(cf_data - torch.Tensor(X_unchange.reshape([1, 3, 224, 224])), 2)
                        # print(dis_loss)
                        vali_loss = myLoss_cf(Xpred1, torch.LongTensor([label_pos_list]).reshape(1).to(self.myDevice))
                        if (predict_label_cf == k):
                            dis_loss_list.append(dis_loss.detach())
                            cf_data_index.append(cf_data.detach())
                            valid_loss_list.append(vali_loss.detach())
                        loss_cf = 0.6* dis_loss + vali_loss 
                        optimizer_cf.zero_grad()
                        loss_cf.backward(retain_graph=True)
                        optimizer_cf.step()
                        count_num = count_num + 1
                    cf_info.setdefault(pos_label_list[k], {"loss": loss_cf.detach(), "dis":dis_loss.detach(), "val": vali_loss.detach(), "dis_track":dis_loss_list, "valid_track":valid_loss_list,  "cf_data":cf_data_index})
                cf_correct_data.append(cf_info)   

           
            cf_label, count, cf_true_correct_list, cf_false_correct_list, y_train_ns_next,  ns_index_for_next, ns_label_for_next , update_ori_for_next, loss_value_noise, cf_mis_ns = cf_search_correct_label(cf_correct_data, ns_data_index, self.noisy_index, self.ori_label, self.ns_label, self.myDevice, self.y)
            
            return cf_true_correct_list, cf_false_correct_list
         
          

    
