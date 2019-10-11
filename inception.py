import tensorflow as tf

from external.nets import inception_v3,inception_v4,inception_resnet_v2

slim = tf.contrib.slim


def inceptionv3(inputs, num_classes, is_training,global_pool ,data_format="channels_first"):
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(inputs,
                                                   num_classes,
                                                   reuse=tf.AUTO_REUSE,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def inceptionv4(inputs, num_classes, is_training,global_pool ,data_format="channels_first"):
  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits, end_points = inception_v4.inception_v4(inputs,
                                                   num_classes,
                                                   reuse=tf.AUTO_REUSE,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions

def inception_resnet(inputs, num_classes, is_training,global_pool ,data_format="channels_first"):
  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs,
                                                   num_classes,
                                                   reuse=tf.AUTO_REUSE)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions