# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:58:03 2018

@author: liguo

为从头训练创建小训练集
"""
import shutil
import os

DIR_PATH = "train_s/"
if not os.path.exists(DIR_PATH):
    os.mkdir(DIR_PATH)
    
for i in range(0, 750):
   name = 'dog.'+str(i)+'.jpg'
   shutil.copyfile('../train/' + name, DIR_PATH + name)
   name = 'cat.'+str(i)+'.jpg'
   shutil.copyfile('../train/' + name, DIR_PATH + name)
