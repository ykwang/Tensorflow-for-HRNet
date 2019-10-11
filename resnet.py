# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""


import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim


def resnet50(inputs, num_classes, is_training, global_pool = True):
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_50(inputs,
                                                num_classes,
                                                global_pool = global_pool,
                                                reuse=tf.AUTO_REUSE)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions

def resnet101(inputs, num_classes, is_training, global_pool = True):
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_101(inputs,
                                                num_classes,
                                                global_pool = global_pool,
                                                reuse=tf.AUTO_REUSE)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions

def resnet152(inputs, num_classes, is_training, global_pool = True):
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_152(inputs,
                                                num_classes,
                                                global_pool = global_pool,
                                                reuse=tf.AUTO_REUSE)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions