"""
@author: liguo

迁移学习的测试
"""

from __future__ import print_function
from nets import nets_factory
from preprocessing import preprocessing_factory
import sys
sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH

import tensorflow as tf

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('infile', 'image/temp.jpg', 'Image file, one image per line.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to testuate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('test_image_size', None, 'test image size.')
FLAGS = tf.app.flags.FLAGS

model_name_to_variables = {'resnet_v1_50':'resnet_v1_50', 'vgg_16':'vgg_16'}
preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path

# 读入图像、预处理模型、网络模型
image_string = tf.placeholder(tf.string) 
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) 
image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

# 数据预处理
if FLAGS.test_image_size is None:
  test_image_size = network_fn.default_image_size
processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
processed_images  = tf.expand_dims(processed_image, 0) 

# 获取输出
logits, _ = network_fn(processed_images)
probabilities = tf.nn.softmax(logits)

# 初始化
init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))
sess = tf.Session()
init_fn(sess)

# 进行推断
content = tf.gfile.FastGFile(FLAGS.infile, 'rb').read()
probs = sess.run(probabilities, feed_dict={image_string:content})
probs = probs[0, 0:]
num = probs[1]

print(num)
