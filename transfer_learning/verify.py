# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:27:32 2018

@author: liguo

迁移学习的验证
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf

from datasets import dataset_utils
from nets import nets_factory
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string('checkpoint_path', 'resnet_v1_50.ckpt',
                           'The directory where the model was written to or an absolute path to a '
                           'checkpoint file.')

tf.app.flags.DEFINE_string('eval_dir', 'verify/data', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string( 'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('dataset_size', 10000, 'The size of dataset.')

tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left '
                           'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

NUM_CLASS = 2

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('必须设置 dataset_dir 参数')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        slim.get_or_create_global_step()
        # 获取验证数据集
        dataset = dataset_utils.get_dataset('verify', FLAGS.dataset_dir, FLAGS.dataset_size, NUM_CLASS) 
        provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset, shuffle=False, common_queue_capacity=2 * FLAGS.batch_size, common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])

        # 获取网络和预处理后的图像
        network_fn = nets_factory.get_network_fn(
                        FLAGS.model_name, num_classes=dataset.num_classes, is_training=False)
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        image = vgg_preprocessing.preprocess_image(
                image, eval_image_size, eval_image_size, is_training=False)

        images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

        # 得到网络的输出
        network_fn = nets_factory.get_network_fn(
                        FLAGS.model_name, num_classes=dataset.num_classes, is_training=False)
        logits, _ = network_fn(images)

        variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # 定义评价指标
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
                'Recall_5': slim.metrics.streaming_recall_at_k(
                        logits, labels, 5),
        })
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            
        num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('\r>> Evaluating %s' % checkpoint_path)
        
        # 开始验证
        start_time = time.time()
        slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore)
        print('total time cost = %.2f' %(time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()
