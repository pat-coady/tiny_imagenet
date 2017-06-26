"""
Tiny ImageNet Model: Logistic Regression
Written by Patrick Coady (pcoady@alum.mit.edu)

Logistic regression NN baseline.
"""
import tensorflow as tf


def logistic_regression(training_batch, config):
  """Baseline logistic regression

  Args:
    training_batch: batch of images (N, 64, 64, 3)
    config: training configuration object

  Returns:
    logits: class prediction scores
  """
  img = tf.cast(training_batch, tf.float32)
  out = (img - 128.0) / 128.0
  out = tf.reshape(out, (config.batch_size, -1))

  with tf.variable_scope('logistic_r',
                         initializer=tf.random_normal_initializer(stddev=0.1 / (56 * 56 * 3) ** 0.5),
                         dtype=tf.float32):
    w = tf.get_variable(shape=(56 * 56 * 3, 200), name='W')
    b = tf.get_variable(shape=(1, 200), name='b')
  logits = tf.matmul(out, w) + b
  tf.summary.histogram('logits', logits)

  l2_loss = tf.nn.l2_loss(w)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)

  return logits
