## Tiny ImageNet: Metrics

import tensorflow as tf

def accuracy(logits, labels):
  """Return batch accuracy

  Args:
    logits: logits (N, C) C = number of classes
    labels: tf.uint8 labels {0 .. 199}

  Returns:
    losses: mean cross entropy loss

  """
  labels = tf.cast(labels, tf.int64)
  pred = tf.argmax(logits, axis=1)

  acc = tf.contrib.metrics.accuracy(pred, labels)
  tf.summary.scalar('acc', acc)

  return acc