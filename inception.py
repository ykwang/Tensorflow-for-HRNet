# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""


import tensorflow as tf

from external.nets import inception_v1,inception_v3,inception_v4,inception_resnet_v2

slim = tf.contrib.slim


def inception_v1(inputs, num_classes, is_training,global_pool =True,data_format="channels_first"):
  with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    logits, end_points = inception_v1.inception_v1(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def inception_v3(inputs, num_classes, is_training,global_pool=True ,data_format="channels_first"):
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions


def inception_v4(inputs, num_classes, is_training,global_pool ,data_format="channels_first"):
  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits, end_points = inception_v4.inception_v4(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions


def inception_resnet_v2(inputs, num_classes, is_training,global_pool ,data_format="channels_first"):
  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions
