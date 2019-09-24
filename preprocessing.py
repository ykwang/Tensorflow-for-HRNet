# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:44:31 2018
@author: shirhe-lyh
Preprocessing images.
Copied and Modified from:
    https://github.com/tensorflow/models/blob/master/research/slim/
    preprocessing/vgg_preprocessing.py
"""

import math
import tensorflow as tf

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512




def _cutout_one(image,mask_size=60, p=0.8, cutout_inside=True, mask_color=0):

  mask_size_half = mask_size // 2
  offset = 1 if mask_size % 2 == 0 else 0

  if np.random.random() > p:
    return image

  h, w = image.shape[:2]
  if cutout_inside:
      cxmin, cxmax = mask_size_half, w + offset - mask_size_half
      cymin, cymax = mask_size_half, h + offset - mask_size_half
  else:
      cxmin, cxmax = 0, w + self.offset
      cymin, cymax = 0, h + self.offset

  cx = np.random.randint(cxmin, cxmax)
  cy = np.random.randint(cymin, cymax)
  xmin = cx - mask_size_half
  ymin = cy - mask_size_half
  xmax = xmin + mask_size
  ymax = ymin + mask_size
  xmin = max(0, xmin)
  ymin = max(0, ymin)
  xmax = min(w, xmax)
  ymax = min(h, ymax)
  image[ymin:ymax, xmin:xmax] = mask_color
  return image

def _cutout(image_list,mask_size=60, p=0.8, cutout_inside=True, mask_color=0):
  outputs = []
  for image in image_list:
     outputs.append(_cutout_one(image,mask_size,p,cutout_inside,mask_color))
  return outputs


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalizes an image."""
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)


def _random_rotate(image, rotate_prob=0.5, rotate_angle_max=30, 
                   interpolation='BILINEAR'):
    def _rotate():
        rotate_angle = tf.random_uniform([], minval=-rotate_angle_max,
                                         maxval=rotate_angle_max, 
                                         dtype=tf.float32)
        rotate_angle = tf.div(tf.multiply(rotate_angle, math.pi), 180.)
        rotated_image = tf.contrib.image.rotate([image], [rotate_angle],
                                                interpolation=interpolation)
        return tf.squeeze(rotated_image)
    
    rand = tf.random_uniform([], minval=0, maxval=1)
    return tf.cond(tf.greater(rand, rotate_prob), lambda: image, _rotate)


def _border_expand(image, mode='CONSTANT', constant_values=255):
    
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    
    def _pad_left_right():
        pad_left = tf.floordiv(height - width, 2)
        pad_right = height - width - pad_left
        return [[0, 0], [pad_left, pad_right], [0, 0]]
        
    def _pad_top_bottom():
        pad_top = tf.floordiv(width - height, 2)
        pad_bottom = width - height - pad_top
        return [[pad_top, pad_bottom], [0, 0], [0, 0]]
    
    paddings = tf.cond(tf.greater(height, width),
                       _pad_left_right,
                       _pad_top_bottom)
    expanded_image = tf.pad(image, paddings, mode=mode, 
                          constant_values=constant_values)
    return expanded_image


def border_expand(image, mode='CONSTANT', constant_values=255,
                  resize=False, output_height=None, output_width=None,
                  channels=3):
    expanded_image = _border_expand(image, mode, constant_values)
    if resize:
        if output_height is None or output_width is None:
            raise ValueError('`output_height` and `output_width` must be '
                             'specified in the resize case.')
        expanded_image = _fixed_sides_resize(expanded_image, output_height,
                                             output_width)
        expanded_image.set_shape([output_height, output_width, channels])
    return expanded_image
        

def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def _fixed_sides_resize(image, output_height, output_width):
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX,
                         border_expand=False, normalize=False,cutout=False,
                         preserving_aspect_ratio_resize=True):
  image = _random_rotate(image, rotate_angle_max=20)
  if border_expand:
      image = _border_expand(image)
  if preserving_aspect_ratio_resize:
      resize_side = tf.random_uniform(
          [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

      image = _aspect_preserving_resize(image, resize_side)
  else:
      image = _fixed_sides_resize(image, resize_side_min, resize_side_min)
  image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  if normalize:
      image =  _normalize(image)
  else:
      image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  if cutout:
      image = _cutout_one(image)
  return image


def preprocess_for_eval(image, output_height, output_width, resize_side,
                        border_expand=False, normalize=False,
                        preserving_aspect_ratio_resize=True):
  if border_expand:
      image = _border_expand(image)
  if preserving_aspect_ratio_resize:
      image = _aspect_preserving_resize(image, resize_side)
  else:
      image = _fixed_sides_resize(image, resize_side, resize_side)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  if normalize:
      return _normalize(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX,
                     border_expand=False, normalize=False,cutout=False,
                     preserving_aspect_ratio_resize=True):
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                                resize_side_min, resize_side_max,
                                border_expand, normalize,cutout,
                                preserving_aspect_ratio_resize)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min, border_expand, normalize,
                               preserving_aspect_ratio_resize)
    
    
def preprocess_images(images, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX,
                     border_expand=False, normalize=False,cutout=False,
                     preserving_aspect_ratio_resize=True):
    images = tf.cast(images, tf.float32)
    def _preprocess_image(image):
        return preprocess_image(image, output_height, output_width,
                                is_training, resize_side_min,
                                resize_side_max, border_expand, normalize,cutout,
                                preserving_aspect_ratio_resize)
    return tf.map_fn(_preprocess_image, elems=images)