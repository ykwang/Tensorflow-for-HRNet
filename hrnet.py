# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""

import tensorflow as tf

import hrnet_v1


def hr_resnet18(inputs, num_classes, is_training,global_pool=True ,data_format="channels_first"):
  with slim.arg_scope(hrnet_v1.hrnet_v1_arg_scope()):
  	hr_resnet18 = hrnet_v1.hr_resnet18(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet48.forward(inputs,num_classes)
    
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def hr_resnet48(inputs, num_classes, is_training,global_pool=True,data_format="channels_first"):
  with slim.arg_scope(hrnet_v1.hrnet_v1_arg_scope()):
  	hr_resnet48 = hrnet_v1.hr_resnet48(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet48.forward(inputs,num_classes)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def hr_resnet64(inputs, num_classes, is_training,global_pool=True ,data_format="channels_first"):
  with slim.arg_scope(hrnet_v1.hrnet_v1_arg_scope()):
  	hr_resnet48 = hrnet_v1.hr_resnet64(pretrained=False, is_training=is_training)
    logits, end_points = hr_resnet64.forward(inputs,num_classes)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions