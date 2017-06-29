"""
Tiny ImageNet Model
Written by Patrick Coady (pcoady@alum.mit.edu)

Architecture is based on VGG-16 model, but the final pool-conv-conv-conv-pool
layers were discarded. The input to the network is a 56x56 RGB crop (versus
224x224 crop for the original VGG-16 model). L2 regularization is applied to
all layer weights. And dropout is applied to the first 2 fully-connected
layers.

1. conv-conv-maxpool
2. conv-conv-maxpool
3. conv-conv-maxpool
4. conv-conv-conv-maxpool
4. fc-4096 (ReLU)
5. fc-2048 (ReLU)
6. fc-200
7. softmax
"""
import tensorflow as tf
import numpy as np


def conv_2d(inputs, filters, kernel_size, name=None):
  """3x3 conv layer: ReLU + (1, 1) stride + He initialization"""

  # He initialization = normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))
  out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                         padding='same', activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                         name=name)
  tf.summary.histogram('act' + name, out)

  return out


def dense_relu(inputs, units, name=None):
  """3x3 conv layer: ReLU + He initialization"""

  # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        name=name)

  tf.summary.histogram('act' + name, out)

  return out


def dense(inputs, units, name=None):
  """3x3 conv layer: ReLU + He initialization"""

  # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                        name=name)
  tf.summary.histogram('act' + name, out)

  return out


def vgg_16_svm(training_batch, config):
  """VGG-like conv-net

  Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object

  Returns:
    class prediction scores
  """
  img = tf.cast(training_batch, tf.float32)
  out = (img - 128.0) / 128.0

  tf.summary.histogram('img', training_batch)
  # (N, 56, 56, 3)
  out = conv_2d(out, 64, (3, 3), 'conv1_1')
  out = conv_2d(out, 64, (3, 3), 'conv1_2')
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

  # (N, 28, 28, 64)
  out = conv_2d(out, 128, (3, 3), 'conv2_1')
  out = conv_2d(out, 128, (3, 3), 'conv2_2')
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

  # (N, 14, 14, 128)
  out = conv_2d(out, 256, (3, 3), 'conv3_1')
  out = conv_2d(out, 256, (3, 3), 'conv3_2')
  out = conv_2d(out, 256, (3, 3), 'conv3_3')
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool3')

  # (N, 7, 7, 256)
  out = conv_2d(out, 512, (3, 3), 'conv4_1')
  out = conv_2d(out, 512, (3, 3), 'conv4_2')
  out = conv_2d(out, 512, (3, 3), 'conv4_3')

  # fc1: flatten -> fully connected layer
  # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
  out = tf.contrib.layers.flatten(out)
  # TODO: Dropout here?
  out = dense_relu(out, 4096, 'fc1')
  out = tf.nn.dropout(out, config.dropout_keep_prob)

  # fc2
  # (N, 4096) -> (N, 2096)
  # TODO: Try dense_tanh
  out = dense_relu(out, 4096, 'fc2')

  # softmax
  # (N, 2048) -> (N, 200)
  logits = dense(out, 200, 'fc3')

  return logits
