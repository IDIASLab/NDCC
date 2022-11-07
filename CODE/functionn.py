#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:02:20 2022

@author: wenting
"""
import numpy as np
import sys
from sklearn.preprocessing import normalize
from random import choice
from random import sample
import random
import torch

from random import choice
def assign_nosie(num_noise, label_list, num_data ):
    pos_label_list = list(set(label_list))
    rand_list = random.sample(range(1, num_data), int(num_noise))
    change_label_index = rand_list
    ori_label = []
    ns_label = []
    for j in range(len(rand_list)):
        true_label = label_list[rand_list[j]]
        ori_label.append(true_label)
        new_list = [item for item in pos_label_list if item not in [true_label]]
        ns_label_indiv = choice(new_list)
        label_list[rand_list[j]] = ns_label_indiv
        ns_label.append(ns_label_indiv)
    return change_label_index, ori_label, ns_label, label_list

def assign_nosie_as(num_noise, label_list, num_data):
    pos_label_list = list(set(label_list)) 
    #generate random data list to convert to noisy
    rand_list = random.sample(range(1, num_data), int(num_noise))
    change_label_index = rand_list
    ori_label = []
    ns_label = []
    for j in range(len(rand_list)):
        true_label = label_list[rand_list[j]]
        ori_label.append(true_label)
        if (true_label != pos_label_list[-1]):
            ns_label_indiv = true_label + 1
        else:
            ns_label_indiv = pos_label_list[0]
        label_list[rand_list[j]] = ns_label_indiv
        ns_label.append(ns_label_indiv)
    return change_label_index, ori_label, ns_label, label_list

def get_index(lst=None, item=''):
  return [i for i in range(len(lst)) if lst[i] == item]

def ns_detect(loss_record_mat_ns, y_train_ns, data_value, epoch):
    ns_data_index = []
    ns_label = []
    ns_output = list(loss_record_mat_ns[:,epoch])
    ns_data_index_temp = get_index(ns_output, 0.0)
    for z in range(len(ns_data_index_temp)):
         if (ns_data_index_temp[z] not in ns_data_index):
            ns_data_index.append(ns_data_index_temp[z])
    ns_data = np.zeros(shape = (len(ns_data_index), 3, 224, 224), dtype = float)
    ns_label = []
    for j in range(len(ns_data_index)):
        da= ns_data_index[j]
        ns_data[j] = data_value[da,:, :]
        ns_label.append(y_train_ns[da])
    return ns_data_index, ns_label, ns_data

def noise_detect_result_analysis(ns_data_index, change_data_index_initial,ori_label_value_initial, ns_label):
    true_detect = []
    wrong_detect = []
    for i in range(len(ns_data_index)):
        ns_data_value_index = ns_data_index[i]
        if (ns_data_value_index in change_data_index_initial):
            true_detect.append(ns_data_value_index)
        else:
            wrong_detect.append(ns_data_value_index)
    miss_detect = [x for x in change_data_index_initial if x not in true_detect]
    return true_detect, wrong_detect, miss_detect
    

def cf_search_correct_label(cf_correct_data, ns_data_index, change_data_index_initial, ori_label_value_initial, ns_label_initial,  myDevice, y_train):
    cf_true_correct_list = []        #record the data index true detect and true correct
    cf_false_correct_list = []      #record the data index wrong detect and true correct
    cf_loss = []
    loss_value_noise = torch.Tensor([0])
    cf_mistook = {}
    
    ns_index_for_next = []
    ns_label_for_next = []
    update_ori_for_next = []
    
    ori_label_dict = {}
    label_dict = {}
    for i in range(len(change_data_index_initial)):
        ori_label_dict.setdefault(change_data_index_initial[i], ori_label_value_initial[i])
        label_dict.setdefault(change_data_index_initial[i], ns_label_initial[i])
        
    ns_ori_label = {}
    for t in range(len(ns_data_index)):
        if (ns_data_index[t] not in  change_data_index_initial):
            ori_label_dict.setdefault(ns_data_index[t], y_train[ns_data_index[t]])
            cf_mistook.setdefault(ns_data_index[t], y_train[ns_data_index[t]])
            
    true_label_list = []
    for j in range(len(ns_data_index)):
        index_temp = ns_data_index[j]
        true_label_list.append(ori_label_dict[index_temp])
    
    for key, value in ori_label_dict.items():
        if key not in ns_data_index:
            ns_index_for_next.append(key)
            update_ori_for_next.append(ori_label_dict[key])
            ns_label_for_next.append(label_dict[key])
            
    y_train_ns_next = y_train
    count = 0  
    cf_label = []
    cf_change_label = []
    
    for z in range(len(cf_correct_data)):
        cf_data_each = cf_correct_data[z]
        data_index_value = ns_data_index[z]
        dis_value_dict = {}  
        for key, value in cf_data_each.items():
            dis_value = value["dis_track"]
            if(len(dis_value) == 0):
                dis_value_dict.setdefault(key, 100)
            else:
                dis_value_dict.setdefault(key, dis_value[-1].numpy())
        find_label = min(dis_value_dict,key=dis_value_dict.get)
        cf_label.append(find_label)
        loss_value = cf_data_each[find_label]["dis_track"]
        y_train_ns_next[data_index_value] = find_label
        loss_value_noise = loss_value_noise.to(myDevice) + loss_value[-1].to(myDevice)
        if(find_label == true_label_list[z]):
            count = count + 1
            cf_true_correct_list.append(data_index_value)
        else:
            ns_index_for_next.append(data_index_value)
            ns_label_for_next.append(find_label)
            true_data = ori_label_dict[data_index_value]
            update_ori_for_next.append(true_data)
            cf_false_correct_list.append(data_index_value)
            
    return cf_label, count, cf_true_correct_list, cf_false_correct_list, y_train_ns_next,  ns_index_for_next, ns_label_for_next, update_ori_for_next, loss_value_noise, cf_mistook

    
    
def label_updated(y_train, ns_data_index, ns_correct_label):
    y_updated = np.zeros(len(y_train), dtype  = int)
    for j in range(len(y_train)):
        if j not in ns_data_index:
            y_updated[j] = y_train[j]
        else:
            c_value_index = ns_data_index.index(j)
            y_updated[j] =  ns_correct_label[c_value_index]
    return y_updated

def noisy_number_modify(change_data_index, ori_label_value, wrong_detect, miss_detect, true_detect, ns_data_index,  cf_correct_label, cf_correct_list, cf_wrong_detect_correct_list):
    cf_change_data_index_updated = []
    cf_ori_data_index_updated = []
    for i in range(len(miss_detect)):
        miss_detect_data_index = miss_detect[i]
        cf_change_data_index_updated.append(miss_detect_data_index)
        label_index_value = change_data_index.index(miss_detect_data_index)
        cf_ori_data_index_updated.append(ori_label_value[label_index_value])
    for j in range(len(wrong_detect)):
        wrong_detect_data_index = wrong_detect[j]
        if (wrong_detect_data_index not in cf_wrong_detect_correct_list):
            cf_change_data_index_updated.append(wrong_detect_data_index)
            wrong_label_index_value = ns_data_index.index(wrong_detect_data_index)
            cf_ori_data_index_updated.append(ori_label_value[wrong_label_index_value])
    for z in range(len(true_detect)):
        true_detect_data_index = true_detect[z]
        if (true_detect_data_index not in cf_correct_list):
            cf_change_data_index_updated.append(true_detect_data_index)
            true_label_index_value = ns_data_index.index(true_detect_data_index)
            cf_ori_data_index_updated.append(ori_label_value[true_label_index_value])
    return cf_change_data_index_updated, cf_ori_data_index_updated
        
        
def ns_data_update(data_sample, change_data_index_updated,  y_train_ns):
    ns_label = []
    ns_data = np.zeros(shape = (len(change_data_index_updated), 3, 224, 224), dtype = float)
    for j in range(len(change_data_index_updated)):
        da = change_data_index_updated[j]
        ns_data[j] = data_sample[da]
        ns_label.append(y_train_ns[j])
    return ns_data, ns_label
