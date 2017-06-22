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