from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
import model
import preprocessing

slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string('images_dir',
                    '',
                    'Path to images (directory).')

flags.DEFINE_string('annotation_path',
                    'annotations.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path', './val_result.json', 'Path to output file.')
flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('backbone','resnet50',
                    'The basic model')
flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
flags.DEFINE_integer('num_classes', 1010, 'Number of classes')
flags.DEFINE_integer('batch_size', 48, 'Batch size')
flags.DEFINE_integer('image_width', 224, 'width')
flags.DEFINE_integer('image_height', 224, 'height')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

FLAGS = tf.app.flags.FLAGS


def get_record_dataset(record_path,
                       reader=None,
                       num_samples=50000,
                       num_classes=7):
    """Get a tensorflow record file.

    Args:

    """
    if not reader:
        reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    labels_to_name_dict = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer.'}
    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=record_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height, width, num_classes):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])

    # Border expand and resize
    image = preprocessing.preprocess_images(image, output_height=height, output_width=width,
                                            border_expand=True, normalize=True
                                            )

    inputs, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    allow_smaller_final_batch=True)
    labels = slim.one_hot_encoding(labels, num_classes)
    return inputs, labels

def main(_):
    if not FLAGS.dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')

    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    images_dir = FLAGS.images_dir
    annotation_path = FLAGS.annotation_path

    num_samples = FLAGS.num_samples
    dataset = get_record_dataset(FLAGS.record_path, num_samples=num_samples,
                                 num_classes=FLAGS.num_classes)
    num_batches = FLAGS.batch_size
    inputs, labels = load_batch(dataset, num_batches, FLAGS.image_width, FLAGS.image_height, FLAGS.num_classes)

    cls_model = model.Model(is_training=False, num_classes=FLAGS.num_classes,
                            backbone=FLAGS.backbone)

    prediction_dict = cls_model.predict(inputs)

    postprocessed_dict = cls_model.postprocess(prediction_dict)

    logits = postprocessed_dict['logits']
    predictions = postprocessed_dict['classes']
    labels = tf.squeeze(labels)

    variables_to_restore = slim.get_variables_to_restore()

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    checkpoint_path = FLAGS.checkpoint_path

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()