# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# encoding=utf8
'''
download_and_convert_mnist.py
'''
import gzip
import os
import sys

import numpy as np
import tensorflow as tf
import scipy
from datasets import dataset_utils


_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]



def _read_images_lables(image_dir):
  images = []
  labels = []
  num_images = 0
  for label in range(10):
    cur_dir = os.path.join(image_dir, str(label))
    for filename in os.listdir(cur_dir):
      if filename[0] == '.':
        continue
      img_file = os.path.join(cur_dir, filename)
      img = scipy.ndimage.imread(img_file, _NUM_CHANNELS==1)
      images.append(img)
      labels.append(label)
      num_images += 1
  labels = np.array(labels)
  images = np.array(images).reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  return images, labels, num_images
    
      

def _add_to_tfrecord(image_dir, tfrecord_writer):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
  images, labels, num_images = _read_images_lables(image_dir)

  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)
    idx = np.array(range(num_images))
    np.random.shuffle(idx)
    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()
        jj = idx[j]
        png_string = sess.run(encoded_png, feed_dict={image: images[jj]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[jj])
        tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_%s.tfrecord' % (dataset_dir, split_name)



def run(dataset_dir, image_dir_train, image_dir_test):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    _add_to_tfrecord(image_dir_train, tfrecord_writer)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    _add_to_tfrecord(image_dir_test, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the MNIST dataset!')
