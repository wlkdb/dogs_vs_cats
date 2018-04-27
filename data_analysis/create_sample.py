# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 05:11:38 2018

@author: liguo

从训练集和测试集中选取部分样本，以进行特征观察
"""

import os
import shutil 
import random
 
TRAIN_PATH = "../train/"
TEST_PATH = "../test/"
SAMPLE_PATH = "sample/"
TRAIN_SAMPLE_PATH = SAMPLE_PATH + "train/"
TRAIN_CAT_SAMPLE_PATH = TRAIN_SAMPLE_PATH + "cat/"
TRAIN_DOG_SAMPLE_PATH = TRAIN_SAMPLE_PATH + "dog/"
TEST_SAMPLE_PATH = SAMPLE_PATH + "test/"
SAMPLE_SIZE = 50

def checkDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
checkDir(SAMPLE_PATH) 
checkDir(TRAIN_SAMPLE_PATH) 
checkDir(TRAIN_CAT_SAMPLE_PATH) 
checkDir(TRAIN_DOG_SAMPLE_PATH) 
checkDir(TEST_SAMPLE_PATH) 
        
# 创建训练集样本
files = os.listdir(TRAIN_PATH)
random.shuffle(files)
train_cat_sample_num = 0
train_dog_sample_num = 0
i = 0
while train_cat_sample_num < SAMPLE_SIZE or train_dog_sample_num < SAMPLE_SIZE:
    if 'cat' in files[i] and train_cat_sample_num < SAMPLE_SIZE:
        shutil.copyfile(TRAIN_PATH + files[i], TRAIN_CAT_SAMPLE_PATH + files[i])
        train_cat_sample_num += 1
    if 'dog' in files[i] and train_dog_sample_num < SAMPLE_SIZE:
        shutil.copyfile(TRAIN_PATH + files[i], TRAIN_DOG_SAMPLE_PATH + files[i])
        train_dog_sample_num += 1
    i += 1
    
# 创建测试集样本
files = os.listdir(TEST_PATH)
random.shuffle(files)
for i in range(SAMPLE_SIZE*2):
    shutil.copyfile(TEST_PATH + files[i], TEST_SAMPLE_PATH + files[i])
        

        
        