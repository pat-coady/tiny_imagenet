"""
Tiny ImageNet Model: Single Layer NN
Written by Patrick Coady (pcoady@alum.mit.edu)

Single-layer NN baseline.
"""

import tensorflow as tf


def single_layer_nn(training_batch, config):
  """Baseline single layer NN 

  Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """
  img = tf.cast(training_batch, tf.float32)
  out = (img - 128.0) / 128.0
  x = tf.reshape(out, (config.batch_size, -1))

  with tf.variable_scope('hid1',
                         initializer=tf.random_normal_initializer(stddev=0.1 /
                                                                  (56 * 56 * 3) ** 0.5),
                         dtype=tf.float32):
    w1 = tf.get_variable(shape=(56 * 56 * 3, 1024), name='W1')
    b1 = tf.get_variable(shape=(1, 1024), name='b')
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
  tf.summary.histogram('hid1', h1)

  with tf.variable_scope('output',
                         initializer=tf.random_normal_initializer(stddev=0.1 /
                                                                  1024 ** 0.5),
                         dtype=tf.float32):
    w2 = tf.get_variable(shape=(1024, 200), name='W2')
    b2 = tf.get_variable(shape=(1, 200), name='b2')
  logits = tf.matmul(h1, w2) + b2
  tf.summary.histogram('logits', logits)

  l2_loss = (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)

  return logits
