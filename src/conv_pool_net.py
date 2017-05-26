## Tiny ImageNet Model: convnet

import tensorflow as tf

def conv_conv_pool(x, chan_in, chan_out, config, name):
  with tf.variable_scope(name+'_conv1',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_in)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_in, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv1 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
  relu1 = tf.nn.relu(conv1 + b)
  if config.dropout:
    relu1 = tf.nn.dropout(relu1, config.dropout_keep_prob)
  tf.summary.histogram(name+'_conv1', relu1)

  with tf.variable_scope(name+'_conv2',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_out)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_out, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b2')

  conv2 = tf.nn.conv2d(relu1, kernel, [1, 1, 1, 1], 'SAME')
  relu2 = tf.nn.relu(conv2 + b)
  if config.dropout:
    relu2 = tf.nn.dropout(relu2, config.dropout_keep_prob)
  tf.summary.histogram(name+'_conv2', relu1)

  y = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  tf.summary.histogram(name+'_maxpool', y)

  return y


def conv_pool_net(training_batch, config):
  """All convolution convnet

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """

  tf.summary.histogram('img', training_batch)
  # in: (N, 64, 64, 3), out: (N, 32, 32, 32)
  ccp1 = conv_conv_pool(training_batch, 3, 32, config, 'stack_1')
  # in: (N, 32, 32, 32), out: (N, 16, 16, 64)
  ccp2 = conv_conv_pool(ccp1, 32, 64, config, 'stack_2')
  # in: (N, 16, 16, 64), out: (N, 8, 8, 128)
  ccp3 = conv_conv_pool(ccp2, 64, 128, config, 'stack_3')

  # fc1: flatten -> fully connected layer, width = 1024
  # (N, 8, 8, 128) -> (N, 8192) -> (N, 2048)
  with tf.variable_scope('fclayer1',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 8192) ** 0.5),
                         dtype=tf.float32):
    wfc1 = tf.get_variable(shape=(8192, 2048), name='wfc1')
    bfc1 = tf.get_variable(shape=(2048,), name='bfc1')

  flat1 = tf.reshape(ccp3, shape=(-1, 8192))
  fc1 = tf.nn.relu(tf.matmul(flat1, wfc1) + bfc1)
  tf.summary.histogram('fc1', fc1)

  # fc2
  # (N, 2048) -> (N, 1024)
  with tf.variable_scope('fclayer2',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 2048) ** 0.5),
                         dtype=tf.float32):
    wfc2 = tf.get_variable(shape=(2048, 1024), name='wfc2')
    bfc2 = tf.get_variable(shape=(1024,), name='bfc2')
  fc2 = tf.nn.relu(tf.matmul(fc1, wfc2) + bfc2)
  tf.summary.histogram('fc2', fc2)

  # fc3
  # (N, 1024) -> (N, 200)
  with tf.variable_scope('fclayer3',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 1024) ** 0.5),
                         dtype=tf.float32):
    wfc3 = tf.get_variable(shape=(1024, 200), name='wfc3')
    bfc3 = tf.get_variable(shape=(200,), name='bfc3')

  logits = tf.matmul(fc2, wfc3) + bfc3
  tf.summary.histogram('logits', logits)

  return logits
