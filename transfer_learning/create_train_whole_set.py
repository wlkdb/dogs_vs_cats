# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:58:03 2018

@author: hasee

为迁移学习创建全训练集
"""
import shutil
import os

def checkDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
checkDir('train/')
checkDir('train/data')
checkDir('train/data/dog/')
checkDir('train/data/cat/')

for i in range(0, 12500):
   name = 'dog.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'train/data/dog/'+name)
   name = 'cat.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'train/data/cat/'+name)
  
