# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""

"""Train a CNN classification model via pretrained TResNet-50 model.
Example Usage:
---------------
python3 train.py \
    --checkpoint_path: Path to pretrained ResNet-50 model.
    --record_path: Path to training tfrecord file.
    --logdir: Path to log directory.
"""

import os
import tensorflow as tf
from tensorflow.python.training import saver as saver_lib

import model
import preprocessing

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path','./training', 
                    '')
flags.DEFINE_string('checkpoint_path', './checkpoint',
                    '')
flags.DEFINE_string('logdir', './training/log', 'Path to log directory.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
flags.DEFINE_float(
    'num_epochs_per_decay', 3.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
flags.DEFINE_integer('num_samples', 32739, 'Number of samples.')
flags.DEFINE_integer('num_classes', 61, 'Number of classes')
flags.DEFINE_integer('num_steps', 10000, 'Number of steps.')
flags.DEFINE_integer('batch_size', 48, 'Batch size')

FLAGS = flags.FLAGS


def get_record_dataset(record_path,
                       reader=None, 
                       num_samples=50000, 
                       num_classes=7):
    """Get a tensorflow record file.
    
    Args:
        
    """
    if not reader:
        reader = tf.TFRecordReader
        
    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = None

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset
    
    
def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        num_samples_per_epoch: he number of samples in each epoch of training.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    
    
def get_init_fn():
    """Returns a function run by che chief worker to warm-start the training.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None
    
    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.logdir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists ' +
            'in %s' % FLAGS.logdir)
        return None
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
    variables_to_restore = slim.get_variables_to_restore()
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)

def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):

    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    
    # Border expand and resize
    image = preprocessing.preprocessing_fn(image, height, width, is_training)
        
    inputs, labels = tf.train.batch([image, label],
                                    batch_size=FLAGS.batch_size,
                                    #capacity=5*FLAGS.batch_size,
                                    allow_smaller_final_batch=True)

def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    num_samples = FLAGS.num_samples
    dataset = get_record_dataset(FLAGS.record_path, num_samples=num_samples, 
                                 num_classes=FLAGS.num_classes)
    
    inputs, labels = load_batch(dataset, FLAGS.batch_size)
    cls_model = model.Model(is_training=True, num_classes=FLAGS.num_classes)
    
    one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes)

    prediction_dict = cls_model.predict(inputs)
    loss_dict = cls_model.loss(prediction_dict, one_hot_labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, one_hot_labels)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)

    global_step = slim.create_global_step()
    learning_rate = configure_learning_rate(num_samples, global_step)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                           momentum=0.9)
#    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)
    tf.summary.scalar('learning_rate', learning_rate)
    
    saver = saver_lib.Saver()

    init_fn = get_init_fn()
    
    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        init_fn=init_fn, number_of_steps=FLAGS.num_steps,
                        save_summaries_secs=20, saver=saver,
                        save_interval_secs=600)
    
if __name__ == '__main__':
    tf.app.run()