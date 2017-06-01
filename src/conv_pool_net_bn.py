## Tiny ImageNet Model: convnet
#
# 1. conv-conv-conv-maxpool (ReLU)
# 2. conv-conv-maxpool (ReLU)
# 3. conv-conv-maxpool (ReLU)
# 4. fc-2048 (ReLU)
# 5. fc-2048 (ReLU)
# 6. fc-1024 (ReLU)
# 7. softmax-200


import tensorflow as tf

def conv_conv_conv_pool(x, chan_in, chan_out, name, config):
  with tf.variable_scope(name+'_conv1',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_in)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_in, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv1 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
  conv1 = tf.layers.batch_normalization(conv1, training=config.training)
  relu1 = tf.nn.relu(conv1 + b)
  tf.summary.histogram(name+'_conv1', relu1)
  tf.summary.scalar(name+'ssq_kernel1', tf.reduce_sum(tf.square(kernel)))
  tf.summary.scalar(name + 'dead_kernel1', tf.reduce_mean(
    tf.cast(tf.less(conv1+b, 0), tf.float32)))


  with tf.variable_scope(name+'_conv2',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_out)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_out, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv2 = tf.nn.conv2d(relu1, kernel, [1, 1, 1, 1], 'SAME')
  conv2 = tf.layers.batch_normalization(conv2, training=config.training)
  relu2 = tf.nn.relu(conv2 + b)
  tf.summary.histogram(name+'_conv2', relu2)
  tf.summary.scalar(name + 'ssq_kernel2', tf.reduce_sum(tf.square(kernel)))
  tf.summary.scalar(name + 'dead_kernel2', tf.reduce_mean(
    tf.cast(tf.less(conv2 + b, 0), tf.float32)))

  with tf.variable_scope(name+'_conv3',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_out)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_out, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv3 = tf.nn.conv2d(relu2, kernel, [1, 1, 1, 1], 'SAME')
  conv3 = tf.layers.batch_normalization(conv3, training=config.training)
  relu3 = tf.nn.relu(conv3 + b)
  tf.summary.histogram(name+'_conv3', relu3)
  tf.summary.scalar(name + 'ssq_kernel3', tf.reduce_sum(tf.square(kernel)))
  tf.summary.scalar(name + 'dead_kernel3', tf.reduce_mean(
    tf.cast(tf.less(conv3 + b, 0), tf.float32)))

  y = tf.nn.max_pool(relu3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  tf.summary.histogram(name+'_maxpool', y)

  return y


def conv_conv_pool(x, chan_in, chan_out, name, config):
  with tf.variable_scope(name+'_conv1',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_in)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_in, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv1 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
  conv1 = tf.layers.batch_normalization(conv1, training=config.training)
  relu1 = tf.nn.relu(conv1 + b)
  tf.summary.histogram(name+'_conv1', relu1)
  tf.summary.scalar(name + 'ssq_kernel1', tf.reduce_sum(tf.square(kernel)))
  tf.summary.scalar(name + 'dead_kernel1', tf.reduce_mean(
    tf.cast(tf.less(conv1 + b, 0), tf.float32)))

  with tf.variable_scope(name+'_conv2',
                         initializer=tf.truncated_normal_initializer(stddev=(2.0 /
                                 (3 * 3 * chan_out)) ** 0.5),
                         dtype=tf.float32):
    kernel = tf.get_variable(shape=(3, 3, chan_out, chan_out), name='kernel')
    b = tf.get_variable(shape=(chan_out,), name='b')

  conv2 = tf.nn.conv2d(relu1, kernel, [1, 1, 1, 1], 'SAME')
  conv2 = tf.layers.batch_normalization(conv2, training=config.training)
  relu2 = tf.nn.relu(conv2 + b)
  tf.summary.histogram(name+'_conv2', relu2)
  tf.summary.scalar(name + 'ssq_kernel2', tf.reduce_sum(tf.square(kernel)))
  tf.summary.scalar(name + 'dead_kernel2', tf.reduce_mean(
    tf.cast(tf.less(conv2 + b, 0), tf.float32)))

  y = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  tf.summary.histogram(name+'_maxpool', y)

  return y


def conv_pool_net_bn(training_batch, config):
  """All convolution convnet

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """

  tf.summary.histogram('img', training_batch)
  # in: (N, 56, 56, 3), out: (N, 28, 28, 32)
  cccp1 = conv_conv_conv_pool(training_batch, 3, 32, 'stack_1', config)
  # in: (N, 28, 28, 32), out: (N, 14, 14, 64)
  ccp2 = conv_conv_pool(cccp1, 32, 64, 'stack_2', config)
  # in: (N, 14, 14, 64), out: (N, 7, 7, 128)
  ccp3 = conv_conv_pool(ccp2, 64, 128, 'stack_3', config)

  # fc1: flatten -> fully connected layer, width = 1024
  # (N, 7, 7, 128) -> (N, 6272) -> (N, 2048)
  with tf.variable_scope('fclayer1',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 6272) ** 0.5),
                         dtype=tf.float32):
    wfc1 = tf.get_variable(shape=(6272, 2048), name='wfc1')
    bfc1 = tf.get_variable(shape=(2048,), name='bfc1')

  flat1 = tf.reshape(ccp3, shape=(-1, 6272))
  fc1 = tf.nn.relu(tf.matmul(flat1, wfc1) + bfc1)
  tf.summary.histogram('fc1', fc1)
  tf.summary.scalar('ssq_fc1', tf.reduce_sum(tf.square(wfc1)))
  tf.summary.scalar('dead_fc1', tf.reduce_mean(
    tf.cast(tf.less(tf.matmul(flat1, wfc1) + bfc1, 0), tf.float32)))
  if config.dropout:
    fc1 = tf.nn.dropout(fc1, config.dropout_keep_prob)

  # fc2
  # (N, 2048) -> (N, 2048)
  with tf.variable_scope('fclayer2',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 2048) ** 0.5),
                         dtype=tf.float32):
    wfc2 = tf.get_variable(shape=(2048, 2048), name='wfc2')
    bfc2 = tf.get_variable(shape=(2048,), name='bfc2')
  fc2 = tf.nn.relu(tf.matmul(fc1, wfc2) + bfc2)
  tf.summary.histogram('fc2', fc2)
  tf.summary.scalar('ssq_fc2', tf.reduce_sum(tf.square(wfc2)))
  tf.summary.scalar('dead_fc2', tf.reduce_mean(
    tf.cast(tf.less(tf.matmul(fc1, wfc2) + bfc2, 0), tf.float32)))
  if config.dropout:
    fc2 = tf.nn.dropout(fc2, config.dropout_keep_prob)

  # fc3
  # (N, 2048) -> (N, 1024)
  with tf.variable_scope('fclayer3',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 2048) ** 0.5),
                         dtype=tf.float32):
    wfc3 = tf.get_variable(shape=(2048, 1024), name='wfc3')
    bfc3 = tf.get_variable(shape=(1024,), name='bfc3')
  fc3 = tf.nn.relu(tf.matmul(fc2, wfc3) + bfc3)
  tf.summary.histogram('fc3', fc3)
  tf.summary.scalar('ssq_fc3', tf.reduce_sum(tf.square(wfc3)))
  tf.summary.scalar('dead_fc3', tf.reduce_mean(
    tf.cast(tf.less(tf.matmul(fc2, wfc3) + bfc3, 0), tf.float32)))
  if config.dropout:
    fc3 = tf.nn.dropout(fc3, config.dropout_keep_prob)

  # softmax
  # (N, 1024) -> (N, 200)
  with tf.variable_scope('softmax',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(2.0 / 1024) ** 0.5),
                         dtype=tf.float32):
    w_sm = tf.get_variable(shape=(1024, 200), name='w_softmax')
    b_sm = tf.get_variable(shape=(200,), name='b_softmax')

  logits = tf.matmul(fc3, w_sm) + b_sm
  tf.summary.histogram('logits', logits)
  tf.summary.scalar('ssq_softmax', tf.reduce_sum(tf.square(w_sm)))

  return logits
