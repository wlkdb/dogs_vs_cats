# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:46:09 2018

@author: liguo
"""

import os
import tensorflow as tf

tf.app.flags.DEFINE_string('train_dir', '../train', 'The directory of train dataset.')

FLAGS = tf.app.flags.FLAGS

# 无法识别的四种类型，需要从训练集中删去
no_cat_or_dog = ["dog.1773", "cat.11184", "cat.4338", "cat.10712", "dog.10747", 
                 "dog.10237", "dog.10801", "cat.5418", "cat.5351", "dog.2614", 
                 "dog.4367", "dog.5604", "dog.8736", "dog.9517", "dog.11299"]
both_cat_and_dog = ["cat.5583", "cat.3822", "cat.9250", "cat.10863", "cat.4688",
                    "cat.11724", "cat.11222", "cat.10266", "cat.9444", "cat.7920", 
                    "cat.7194", "cat.5355", "cat.724", "dog.2461", "dog.8507"]
hard_to_recognition = ["cat.6402", "cat.6987", "dog.11083", "cat.12499", "cat.2753", 
                       "dog.669", "cat.2150", "dog.5490", "cat.12493", "cat.7703", 
                       "dog.3430", "cat.2433", "cat.3250", "dog.4386", "dog.12223", 
                       "cat.9770", "cat.9626", "cat.6649", "cat.5324", "cat.335",
                       "cat.10029", "dog.1835", "dog.3322", "dog.3524", "dog.6921",
                       "dog.7413", "dog.10939", "dog.11248"]
too_abstract_image = ["dog.8898", "dog.1895", "dog.4690", "dog.1308", "dog.10190",
                      "dog.10161"]

# 标注反了，需要修改标注
label_reverse = ["cat.4085", "cat.12272", "dog.2877", "dog.4334", "dog.10401", "dog.10797", 
                 "dog.11731"]

print("no_cat_or_dog = ", len(no_cat_or_dog))
print("both_cat_and_dog = ", len(both_cat_and_dog))
print("hard_to_recognition = ", len(hard_to_recognition))
print("too_abstract_image = ", len(too_abstract_image))
print("")
print("label_reverse = ", len(label_reverse))


def del_image_in_list(name_list):
    for name in name_list:
        path = os.path.join(FLAGS.train_dir, name + ".jpg")
        if os.path.exists(path):
            os.remove(path)

cat_index = 12500
dog_index = 12500

def change_label_in_list(name_list):
    global cat_index
    global dog_index
    for name in name_list:
        path = os.path.join(FLAGS.train_dir, name + ".jpg")
        if os.path.exists(path):
            if "cat" in name:
                new_name = "dog." + str(dog_index)
                dog_index += 1
            else:
                new_name = "cat." + str(cat_index)
                cat_index += 1
            new_path = os.path.join(FLAGS.train_dir, new_name + ".jpg")
            os.rename(path, new_path)
            
if not os.path.exists(FLAGS.train_dir):
    print("error: train directory doesn't exist.")
    exit()

# 删除无法识别的图像
del_image_in_list(no_cat_or_dog)
del_image_in_list(both_cat_and_dog)
del_image_in_list(hard_to_recognition)
del_image_in_list(too_abstract_image)

# 修改标注反了的文件名
change_label_in_list(label_reverse)

print("")
print("Done")









