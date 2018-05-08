# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:58:03 2018

@author: liguo

为迁移学习创建小训练集
"""
import shutil
import os

def checkDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
checkDir('train_s/')
checkDir('train_s/data')
checkDir('train_s/data/dog/')
checkDir('train_s/data/cat/')

num_dog = 0
num_cat = 0
for i in range(0, 5000):
    name = 'dog.'+str(i)+'.jpg'
    if num_dog < 750 and os.path.exists('../train/'+name):
        shutil.copyfile('../train/'+name, 'train_s/data/dog/'+name)
        num_dog += 1
      
    name = 'cat.'+str(i)+'.jpg'
    if num_cat < 750 and os.path.exists('../train/'+name):
        shutil.copyfile('../train/'+name, 'train_s/data/cat/'+name)
        num_cat += 1
        
    if num_dog >= 750 and num_cat >= 750:
        break
