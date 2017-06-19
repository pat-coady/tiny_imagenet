# Tiny ImageNet Model: VGG-16 Model
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
  return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                          strides=strides, padding='same', activation=tf.nn.relu,
                          kernel_initializer=tf.random_normal_initializer(stddev=stddev))


def dense_relu(inputs, units):
  """3x3 conv layer: ReLU + He initialization"""

  stddev = np.sqrt(2 / int(inputs.shape[1]))  # sqrt(2/fan-in)
  return tf.layers.dense(inputs, units, activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev))


def dense(inputs, units):
  """3x3 conv layer: ReLU + He initialization"""

  stddev = np.sqrt(2 / int(inputs.shape[1]))  # sqrt(2/fan-in)
  return tf.layers.dense(inputs, units,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev))


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

  # (N, 28, 28, 64)
  out = conv_2d(out, 128, (3, 3))
  out = conv_2d(out, 128, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))

  # (N, 14, 14, 128)
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))

  # (N, 7, 7, 256)
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))

  # fc1: flatten -> fully connected layer
  # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
  out = tf.contrib.layers.flatten(out)
  out = dense_relu(out, 4096)
  out = tf.nn.dropout(out, config.dropout_keep_prob)

  # fc2
  # (N, 4096) -> (N, 2048)
  out = dense_relu(out, 2048)
  out = tf.nn.dropout(out, config.dropout_keep_prob)

  # softmax
  # (N, 2048) -> (N, 200)
  logits = dense(out, 200)
  tf.summary.histogram('logits', logits)

  return logits
