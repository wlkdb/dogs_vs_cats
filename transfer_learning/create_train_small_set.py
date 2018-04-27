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

for i in range(0, 750):
   name = 'dog.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'train_s/data/dog/'+name)
   name = 'cat.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'train_s/data/cat/'+name)
