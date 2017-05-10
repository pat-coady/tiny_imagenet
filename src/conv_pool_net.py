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
  stack1 = conv_conv_pool(training_batch, 3, 64, config, 's1')
  stack2 = conv_conv_pool(stack1, 64, 64, config, 's2')

  # 1x1 conv, 16 filter, stride = 1
  # (N, 16, 16, 64) -> (N, 16, 16, 16)
  with tf.variable_scope('conv3',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (1 * 1 * 64)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(1, 1, 64, 16), name='kernel')
    b = tf.get_variable(shape=(16,), name='b')
  conv3 = tf.nn.conv2d(stack2, kernel, [1, 1, 1, 1], 'SAME')
  relu3 = tf.nn.relu(conv3 + b)
  if config.dropout:
    relu3 = tf.nn.dropout(relu3, config.dropout_keep_prob)
  tf.summary.histogram('relu3', relu3)

  # flatten -> fully connected layer, width = 1024
  # (N, 16, 16, 16) -> (N, 4096) -> (N, 1024)
  with tf.variable_scope('fclayer1',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (16 * 16 * 16)) ** 0.5),
                         dtype=tf.float32):
    wfc1 = tf.get_variable(shape=(16 * 16 * 16, 1024), name='wfc1')
    bfc1 = tf.get_variable(shape=(1024,), name='bfc1')

  flat1 = tf.reshape(relu3, shape=(-1, 16 * 16 * 16))
  fc1 = tf.nn.relu(tf.matmul(flat1, wfc1) + bfc1)
  # fc1 = tf.nn.dropout(fc1, config.dropout_keep_prob)
  tf.summary.histogram('fc1', fc1)

  # fully connected layer, width = 200
  # (N, 1024) -> (N, 200)
  with tf.variable_scope('fclayer2',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 1024) ** 0.5),
                         dtype=tf.float32):
    wfc2 = tf.get_variable(shape=(1024, 200), name='wfc2')
    bfc2 = tf.get_variable(shape=(200,), name='bfc2')

  logits = tf.nn.relu(tf.matmul(fc1, wfc2) + bfc2)
  tf.summary.histogram('logits', logits)

  return logits
