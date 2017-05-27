### Tiny ImageNet: Main Training

from conv_pool_net import *
from metrics import *
from losses import *
from input_pipe import *
from datetime import datetime
import numpy as np
import os


class TrainConfig(object):
  """Training configuration"""
  batch_size = 64
  num_epochs = 15
  summary_interval = 10
  save_every = 50
  lr = 0.01
  momentum = 0.9
  dropout = True
  dropout_keep_prob = 0.5
  model_name = 'conv_pool_net'
  model = staticmethod(globals()[model_name])
  experiment_name = 'initial_tune'


def optimizer(loss, config):
  """Add training operation, loss function and global step to Graph.

  Args:
    config: training configuration object
    loss: model loss tensor

  Returns:
    tuple: (training operation, step loss, global step num) 
  """
  global_step = tf.Variable(0, trainable=False, name='global_step')
  optim = tf.train.MomentumOptimizer(config.lr, config.momentum,
                                     use_nesterov=True)
  train_op = optim.minimize(loss, global_step=global_step)

  return train_op, global_step


def get_logdir():
  """Return unique logdir based on datetime"""
  now = datetime.utcnow().strftime("%m%d%H%M%S")
  logdir = "run-{}/".format(now)

  return logdir


def model_wrapper(mode, config):
  """Wrap up: input data queue, regression model and loss functions 

  Args:
    mode: 'train' or 'val
    config: model configuration object

  Returns:
    loss and accuracy tensors
  """
  with tf.device('/cpu:0'):
    imgs, labels = batch_q(mode, config)


  logits = config.model(imgs, config)
  softmax_ce_loss(logits, labels)
  acc = accuracy(logits, labels)
  total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')

  return total_loss, acc


def validate(ckpt_path):
  """Load most recent checkpoint and run on validation set"""
  config = TrainConfig()
  config.dropout = False  # disable dropout for validation
  accs, losses = [], []

  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      loss, acc = model_wrapper('val', config)
      saver = tf.train.Saver()
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
    with tf.Session() as sess:
      init.run()
      saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        iters = 0
        while not coord.should_stop():
          iters += 1
          step_loss, step_acc = sess.run([loss, acc])
          accs.append(step_acc)
          losses.append(step_loss)
          if iters > 20: break
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
      finally:
        coord.request_stop()
        coord.join(threads)
  mean_loss, mean_acc = np.mean(losses), np.mean(accs)
  print('Validation. Loss: {:.3f}, Accuracy: {:.4f}'.
        format(mean_loss, mean_acc))
  config.dropout = True

  return mean_loss, mean_acc


def main():
  config = TrainConfig()
  g = tf.Graph()
  ckpt_path = 'checkpoints/' + config.model_name + '/' + config.experiment_name
  tflog_path = ('tf_logs/' + config.model_name + '/' +
                config.experiment_name + '/' + get_logdir())
  if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
  with g.as_default():
    writer = tf.summary.FileWriter(tflog_path, g)
    loss, acc = model_wrapper('train', config)
    train_op, g_step = optimizer(loss, config)
    val_acc = tf.Variable(0.0, trainable=False)
    val_loss = tf.Variable(0.0, trainable=False)
    tf.summary.scalar('val_loss', val_loss)
    tf.summary.scalar('val_accuracy', val_acc)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    [tf.summary.histogram(v.name.replace(':', '_'), v)
     for v in tf.trainable_variables()]
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as sess:
      init.run()
      # saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        losses, accs = [], []
        while not coord.should_stop():
          step_loss, _, step, step_acc = sess.run([loss, train_op,
                                                   g_step, acc])
          losses.append(step_loss)
          accs.append(step_acc)
          if step % config.save_every == 0:
            saver.save(sess, ckpt_path, step)
            mean_loss, mean_acc = validate(ckpt_path)
            val_acc.load(mean_acc)
            val_loss.load(mean_loss)
          if step % config.summary_interval == 0:
            writer.add_summary(sess.run(summ), step)
            print('Iteration: {}, Loss: {:.3f}, Accuracy: {:.4f}'.
                  format(step, np.mean(losses), np.mean(accs)))
            losses, accs = [], []
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
      finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  main()
