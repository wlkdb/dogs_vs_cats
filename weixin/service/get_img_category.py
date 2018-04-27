#!/usr/bin/env python

from __future__ import print_function
import sys
sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH
import tensorflow as tf

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('infile', 'img/temp', 'Image file, one image per line.')
tf.app.flags.DEFINE_boolean('tfrecord',False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string('outfile',None, 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

from nets import nets_factory
from preprocessing import preprocessing_factory

#import win_unicode_console 
#win_unicode_console.enable()
allow_soft_placement=True

slim = tf.contrib.slim

model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v2_50':'resnet_v2_50','vgg_16':'vgg_16'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name

model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path

image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files
      
image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size

#print(image)
#image = tf.add(image, tf.zeros([eval_image_size, eval_image_size, 3], tf.uint8), name="input")

processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

processed_images  = tf.expand_dims(processed_image, 0) 
# Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)

index = 0
    
path = FLAGS.infile
#        print(path)
content = tf.gfile.FastGFile(path, 'rb').read()
probs = sess.run(probabilities, feed_dict={image_string:content})

#        print(probs)
probs = probs[0, 0:]
num = probs[1]
print(num)
