"""
Tiny ImageNet: Loss Functions
"""
import tensorflow as tf


def softmax_ce_loss(logits, labels):
  """Softmax + cross-entropy loss

  Args:
    logits: logits (N, C) C = number of classes
    labels: tf.uint8 labels {0 .. 199}

  Returns:
    losses: mean cross entropy loss

  """
  labels = tf.cast(labels, tf.int32)
  ce_loss = tf.losses.sparse_softmax_cross_entropy(labels,
                                                   logits,
                                                   weights=1.0)
  tf.summary.scalar('loss', ce_loss)


def softmax_smooth_ce_loss(logits, labels):
  """Softmax + cross-entropy loss with label smoothing

  Args:
    logits: logits (N, C) C = number of classes
    labels: tf.uint8 labels {0 .. 199}

  Returns:
    losses: mean cross entropy loss

  """
  labels = tf.cast(labels, tf.int32)
  ohe = tf.one_hot(labels, 200, dtype=tf.int32)
  ce_loss = tf.losses.softmax_cross_entropy(ohe,
                                            logits,
                                            label_smoothing=0.1)
  tf.summary.scalar('loss', ce_loss)


def svm_loss(logits, labels):
  """SVM loss

  Args:
    logits: logits (N, C) C = number of classes
    labels: tf.uint8 labels {0 .. 199}

  Returns:
    losses: mean cross entropy loss

  """
  c = 1.0
  labels = tf.cast(labels, tf.int32)
  ohe = tf.one_hot(labels, 200, dtype=tf.float32, on_value=-1.0, off_value=1.0)
  svm_l = c * tf.reduce_sum(tf.maximum(1.0 + ohe * logits, 0.0))
  tf.add_to_collection(tf.GraphKeys.LOSSES, svm_l)

  tf.summary.scalar('loss', svm_l)
