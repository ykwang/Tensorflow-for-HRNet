# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""

import tensorflow as tf

import hrnet_v1

slim = tf.contrib.slim


def hr_resnet18(inputs, num_classes, is_training):
  with slim.arg_scope(hrnet_v1.hrnet_arg_scope()):
    hr_resnet_18 = hrnet_v1.hr_resnet18(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet_18.forward(inputs,num_classes)
    
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def hr_resnet48(inputs, num_classes, is_training):
  with slim.arg_scope(hrnet_v1.hrnet_arg_scope()):
    hr_resnet_48 = hrnet_v1.hr_resnet48(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet_48.forward(inputs, num_classes)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def hr_resnet64(inputs, num_classes, is_training):
  with slim.arg_scope(hrnet_v1.hrnet_arg_scope()):
    hr_resnet_64 = hrnet_v1.hr_resnet64(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet_64.forward(inputs,num_classes)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions