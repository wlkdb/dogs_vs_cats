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

num_dog = 0
num_cat = 0
for i in range(5000, 12600):
    name = 'dog.'+str(i)+'.jpg'
    if num_dog < 5000 and os.path.exists('../train/'+name):
        shutil.copyfile('../train/'+name, 'verify/data/dog/'+name)
        num_dog += 1
      
    name = 'cat.'+str(i)+'.jpg'
    if num_cat < 5000 and os.path.exists('../train/'+name):
        shutil.copyfile('../train/'+name, 'verify/data/cat/'+name)
        num_cat += 1
        
    if num_dog >= 5000 and num_cat >= 5000:
        break
        