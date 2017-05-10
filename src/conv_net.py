## Tiny ImageNet Model: convnet

import tensorflow as tf

def conv_net(training_batch, config):
  """All convolution convnet

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """

  tf.summary.histogram('img', training_batch)
  # 3x3 conv, 64 filters
  # (N, 64, 64, 3) -> (N, 64, 64, 64)
  with tf.variable_scope('conv1',
                         initializer=tf.truncated_normal_initializer(stddev=0.1 /
                                 (3 * 3 * 3) ** 0.5),
                         dtype=tf.float32):
    filter1 = tf.get_variable(shape=(3, 3, 3, 64), name='filter1')
    b1 = tf.get_variable(shape=(64,), name='b1')
  conv1 = tf.nn.conv2d(training_batch, filter1, [1, 1, 1, 1], 'SAME')
  relu1 = tf.nn.relu(conv1 + b1)
  # relu1 = tf.nn.dropout(relu1, config.dropout_keep_prob)
  tf.summary.histogram('relu1', relu1)

  # 3x3 conv, 64 filter, stride = 2
  # (N, 64, 64, 64) -> (N, 32, 32, 64)
  with tf.variable_scope('conv2',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * 64)) ** 0.5),
                         dtype=tf.float32):
    filter2 = tf.get_variable(shape=(3, 3, 64, 64), name='filter2')
    b2 = tf.get_variable(shape=(64,), name='b1')
  conv2 = tf.nn.conv2d(relu1, filter2, [1, 2, 2, 1], 'SAME')
  relu2 = tf.nn.relu(conv2 + b2)
  # relu2 = tf.nn.dropout(relu2, config.dropout_keep_prob)
  tf.summary.histogram('relu2', relu2)

  # 1x1 conv, 32 filter, stride = 1
  # (N, 32, 32, 64) -> (N, 32, 32, 32)
  with tf.variable_scope('conv3',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * 64)) ** 0.5),
                         dtype=tf.float32):
    filter3 = tf.get_variable(shape=(3, 3, 64, 32), name='filter1')
    b3 = tf.get_variable(shape=(32,), name='b3')
  conv3 = tf.nn.conv2d(relu2, filter3, [1, 1, 1, 1], 'SAME')
  relu3 = tf.nn.relu(conv3 + b3)
  # relu3 = tf.nn.dropout(relu3, config.dropout_keep_prob)
  tf.summary.histogram('relu3', relu3)

  # flatten -> fully connected layer, width = 1024
  # (N, 32, 32, 32) -> (N, 3072) -> (N, 1024)
  with tf.variable_scope('fclayer1',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (32*32*32)) ** 0.5),
                         dtype=tf.float32):
    wfc1 = tf.get_variable(shape=(32*32*32, 1024), name='wfc1')
    bfc1 = tf.get_variable(shape=(1024,), name='bfc1')

  flat1 = tf.reshape(relu3, shape=(-1, 32*32*32))
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
