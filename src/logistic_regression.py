## Tiny ImageNet Model: Logistic Regression

import tensorflow as tf

def logistic_regression(training_batch, config):
  """Baseline logistic regression

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """
  x = tf.reshape(training_batch, (config.batch_size, -1))

  with tf.variable_scope('logistic_r',
                         initializer=tf.random_normal_initializer(stddev=0.1 / (64 * 64 * 3) ** 0.5),
                         dtype=tf.float32):
    w = tf.get_variable(shape=(64 * 64 * 3, 200), name='W')
    b = tf.get_variable(shape=(1, 200), name='b')
  logits = tf.matmul(x, w) + b
  tf.summary.histogram('logits', logits)

  weight_loss = tf.nn.l2_loss(w) * config.reg

  tf.add_to_collection('losses', weight_loss)

  return logits





