import os
import numpy as np
import random
import cv2
import time
import tensorflow as tf 

import preprocessing

slim = tf.contrib.slim

class DatasetProvider:
    def __init__(self, data, labels,num_classes,need_shuffle=True,need_label=True,is_training=True):
        self._data = data
        self._indicator = 0
        self._need_shuffle = need_shuffle
        self._need_label = need_label
        self._num_classes = num_classes
        self._is_training = is_training
        if self._need_label:
          self._labels = labels
          assert len(data) == len(labels)
          self._one_hot_labels = self.one_hot(self._labels)
          
        if self._need_shuffle:
            self._shuffle_data()
         
    def _shuffle_data(self):
        p = np.random.permutation(len(self._data))
        self._data = self._data[p]
        if self._need_label:
            self._labels = self._labels[p]
        
    def one_hot(self, label):

        onehot_encoded = slim.one_hot_encoding(label,self._num_classes)
        
        return onehot_encoded
      
    def next_batch(self, batch_size,img_width=224, img_height=224):
      
        while True:
          if self._indicator + batch_size >= len(self._data):
              p = np.random.permutation(len(self._data))
              self._data = self._data[p]
              if self._need_label:
                self._labels = self._labels[p]
              self._indicator = 0
          data_batch = np.zeros((batch_size, img_width, img_height, 3))
          if self._need_label:
            label_batch = np.zeros((batch_size, self._num_classes))

          #print('# _indicator : {}.'.format(self._indicator))
          for i in range(self._indicator, self._indicator + batch_size):
              img_path = self._data[i]
              #print('# Images Path : {}.'.format(img_path))
              img = cv2.imread(img_path)
              img = tf.cast(img,tf.float32)
              img = preprocessing.preprocess_image(
                    img, img_width, img_height, 
                    is_training=self._is_training,
                    border_expand=True, normalize=True,cutout=False,
                    preserving_aspect_ratio_resize=False)

              data_batch[i - self._indicator] = img;
              if self._need_label:
                label_batch[i - self._indicator]  = self._one_hot_labels[i]
          
          self._indicator += batch_size 
          if self._need_label:
            yield data_batch, label_batch
          else:
            yield data_batch