# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:58:03 2018

@author: liguo

为迁移学习创建验证集
"""
import shutil
import os

def checkDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
checkDir('verify/')
checkDir('verify/data')
checkDir('verify/data/dog/')
checkDir('verify/data/cat/')

for i in range(7500, 12500):
   name = 'dog.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'verify/data/dog/'+name)
   name = 'cat.'+str(i)+'.jpg'
   shutil.copyfile('../train/'+name, 'verify/data/cat/'+name)
