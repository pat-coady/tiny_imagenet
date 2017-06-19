## Tiny ImageNet Model: VGG-16 Model
#
# 1. conv-conv-conv-maxpool (ReLU)
# 2. conv-conv-maxpool (ReLU)
# 3. conv-conv-maxpool (ReLU)
# 4. fc-2048 (ReLU)
# 5. fc-2048 (ReLU)
# 6. fc-1024 (ReLU)
# 7. softmax-200


import tensorflow as tf
import numpy as np


def conv_2d(inputs, filters, kernel_size, strides=(1, 1)):
  """3x3 conv layer: ReLU + He initialization"""

  stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))  # sqrt(2/fan-in)
  out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                         strides=strides, padding='same', activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev))

  return out


def vgg_16(training_batch, config):
  """All convolution convnet

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    class prediction scores
  """

  img = tf.cast(training_batch, tf.float32)
  out = (img - 128.0) / 128.0
  # TODO: Normalize and scale based on training set statistics?
  tf.summary.histogram('img', training_batch)
  # (N, 56, 56, 3)
  out = conv_2d(out, 64, (3, 3))
  out = conv_2d(out, 64, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))
  print(out.shape)

  # (N, 28, 28, 64)
  out = conv_2d(out, 128, (3, 3))
  out = conv_2d(out, 128, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))
  print(out.shape)

  # (N, 14, 14, 128)
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))
  print(out.shape)

  # (N, 7, 7, 256)
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))

  # fc1: flatten -> fully connected layer, width = 1024
  # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
  out = tf.contrib.layers.flatten(out)

  with tf.variable_scope('fclayer1',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 25088) ** 0.5),
                         dtype=tf.float32):
    wfc1 = tf.get_variable(shape=(25088, 4096), name='wfc1')
    bfc1 = tf.get_variable(shape=(4096,), name='bfc1')

  out = tf.nn.relu(tf.matmul(out, wfc1) + bfc1)
  tf.summary.histogram('fc1', out)
  tf.summary.scalar('ssq_fc1', tf.reduce_sum(tf.square(out)))
  # tf.summary.scalar('dead_fc1', tf.reduce_mean(
  #   tf.cast(tf.less(tf.matmul(out, wfc1) + bfc1, 0), tf.float32)))
  out = tf.nn.dropout(out, config.dropout_keep_prob)

  # fc2
  # (N, 4096) -> (N, 2048)
  with tf.variable_scope('fclayer2',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 4096) ** 0.5),
                         dtype=tf.float32):
    wfc2 = tf.get_variable(shape=(4096, 2048), name='wfc2')
    bfc2 = tf.get_variable(shape=(2048,), name='bfc2')
  out = tf.nn.relu(tf.matmul(out, wfc2) + bfc2)
  tf.summary.histogram('fc2', out)
  tf.summary.scalar('ssq_fc2', tf.reduce_sum(tf.square(out)))
  # tf.summary.scalar('dead_fc2', tf.reduce_mean(
  #   tf.cast(tf.less(tf.matmul(out, wfc2) + bfc2, 0), tf.float32)))
  out = tf.nn.dropout(out, config.dropout_keep_prob)

  # softmax
  # (N, 2048) -> (N, 200)
  with tf.variable_scope('softmax',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 2048) ** 0.5),
                         dtype=tf.float32):
    w_sm = tf.get_variable(shape=(2048, 200), name='w_softmax')
    b_sm = tf.get_variable(shape=(200,), name='b_softmax')

  logits = tf.matmul(out, w_sm) + b_sm
  tf.summary.histogram('logits', logits)
  tf.summary.scalar('ssq_softmax', tf.reduce_sum(tf.square(w_sm)))

  return logits
