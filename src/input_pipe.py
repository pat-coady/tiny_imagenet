### Tiny ImageNet: Input Pipeline

import glob
import re
import tensorflow as tf
import random


def load_filenames_labels(mode):
  """Gets filenames and labels

  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for train and val)

  Returns:
    filenames: list of tuples: (jpeg filename with path, label)
  """
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob('../tiny-imagenet-200/train/*/images/*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif mode == 'val':
    with open('../tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = '../tiny-imagenet-200/val/images/' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))

  return filenames_labels


def build_label_dicts():
  """Build look-up dictionaries for class label, and class description

  Class labels are 0 to 199 in the same order as 
    tiny-imagenet-200/wnids.txt. Class text descriptions are from 
    tiny-imagenet-200/words.txt

  Returns:
    label_dict: 
      keys = image directory string (e.g. "n01944390")
      values = class integer {0 .. 199}
    class_desc:
      keys = class integer {0 .. 199}
      values = text description from words.txt
  """
  label_dict, class_description = {}, {}
  with open('../tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('../tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description


def read_image(filename_q, mode):
  """Load next jpeg file from filename / label queue

  Args:
    filename_q: Queue with 2 columns: filename string and label string.
     filename string is relative path to jpeg file. label string is text-
     formatted integer between '0' and '199'
    mode: 'train' or 'val'

  Returns:
    [img, label]: 
      img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
      label = tf.unit8 target class label: {0 .. 199}
  """
  item = filename_q.dequeue()
  filename = item[0]
  label = item[1]
  file = tf.read_file(filename)
  img = tf.image.decode_jpeg(file, channels=3)
  # image distortions: left/right, random hue, random color saturation
  if mode == 'train':
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_hue(img, 0.05)
    img = tf.image.random_saturation(img, 0.5, 2.0)

  img = tf.image.per_image_standardization(img)
  # TODO: Add noise?

  label = tf.string_to_number(label, tf.int32)
  label = tf.cast(label, tf.uint8)

  return [img, label]


def batch_q(mode, config):
  """Return batch of images using filename Queue

  Args:
    mode: 'train' or 'val'
    config: training configuration object

  Returns:
    imgs: tf.uint8 tensor [batch_size, height, width, channels]
    labels: tf.uint8 tensor [batch_size,]

  """
  filenames_labels = load_filenames_labels(mode)
  random.shuffle(filenames_labels)
  if config.num_examples is not None:
    filenames_labels = filenames_labels[:config.num_examples]
  filename_q = tf.train.input_producer(filenames_labels,
                                       num_epochs=config.num_epochs,
                                       shuffle=True)

  return tf.train.batch(read_image(filename_q, mode),
                        config.batch_size, shapes=[(64, 64, 3), ()],
                        capacity=1024, num_threads=4)
