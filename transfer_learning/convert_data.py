# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:51:46 2018

@author: liguo
数据格式转换
"""

import math
import os
import random
import sys
import tensorflow as tf
from datasets import dataset_utils

_RANDOM_SEED = 0
NUM_CLASS = 2 

tf.app.flags.DEFINE_string('dataset_dir', None, 'output TFRecords directory.')
tf.app.flags.DEFINE_integer('type', 0, 'create train or verify dataset.') # 0代表创建训练集，其它代表创建验证集

FLAGS = tf.app.flags.FLAGS
dataset_type = "train" if FLAGS.type == 0 else "verify"

class ImageReader(object):

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_dataset_filename(dataset_dir, class_id):
    output_filename = '%s_%d.tfrecord' % (dataset_type, class_id)
    return os.path.join(dataset_dir, output_filename)

def dataset_exists(dataset_dir):
    for class_id in range(NUM_CLASS):
        output_filename = get_dataset_filename(dataset_dir, class_id)
    if not tf.gfile.Exists(output_filename):
        return False
    return True

def get_filenames_and_classes(dataset_dir):
    dir_root = os.path.join(dataset_dir, 'data')
    directories = []
    class_names = []
    for filename in os.listdir(dir_root):
        path = os.path.join(dir_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
    return photo_filenames, sorted(class_names)

def convert_dataset(filenames, class_names_to_ids, dataset_dir):
    num_per_shard = int(math.ceil(len(filenames) / float(NUM_CLASS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for class_id in range(NUM_CLASS):
                output_filename = get_dataset_filename(dataset_dir, class_id)
                
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = class_id * num_per_shard
                    end_ndx = min((class_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d class %d' % (
                                i+1, len(filenames), class_id))
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        example = dataset_utils.image_to_tfexample(
                                image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()
    
def main(_):
    dataset_dir = FLAGS.dataset_dir
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    
    if dataset_exists(dataset_dir):
        print('error: 数据集已存在')
        exit()
    
    photo_filenames, class_names = get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    
    # 数据集转换
    convert_dataset(photo_filenames, class_names_to_ids, dataset_dir)
    
    # 保存类别标签
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    
    print('\nDone')
    
if __name__ == '__main__':
  tf.app.run()
