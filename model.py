# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""

import tensorflow as tf

from tensorflow.contrib.slim import nets

import preprocessing
from resnet import resnet32,resnet50,resnet101,resnet152
from mobilenet import mobilenet
from inception import inception_v3,inception_v4,inception_resnet_v2
from hrnet import hr_resnet18,hr_resnet48,hr_resnet64

slim = tf.contrib.slim

backbone_fn = {'resnet32': resnet32,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152,
                'inception_v3': inception_v3,
                'inception_v4':inception_v4,
                'inception_resnet_v2':inception_resnet_v2,
                'mobilenet':mobilenet,
                'hr_resnet18': hr_resnet18,
                'hr_resnet48': hr_resnet48,
                'hr_resnet64': hr_resnet64}

        
class Model(object):
    
    def __init__(self, num_classes, is_training,backbone):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        self._num_classes = num_classes
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size
        self._backbone = backbone
        
    @property
    def num_classes(self):
        return self._num_classes
        
    def preprocess(self, inputs,fixed_resize_side=224,
                 default_image_size=224):
        """preprocessing.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, default_image_size, default_image_size, 
            resize_side_min=fixed_resize_side,
            is_training=self._is_training,
            border_expand=True, normalize=True,cutout=True,
            preserving_aspect_ratio_resize=True)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        logits, endpoints = backbone_fn[self._backbone](
                preprocessed_inputs, num_classes=self.num_classes,
                is_training=self._is_training)
        
        prediction_dict = {'logits': logits}
        return prediction_dict
    
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.argmax(logits, axis=1)
        postprocessed_dict = {'logits': logits,
                              'classes': classes}
        return postprocessed_dict
    
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits, 
            labels=groundtruth_lists,
            scope='Loss')
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
        
    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            accuracy: The scalar accuracy.
        """
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy