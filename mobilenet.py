import tensorflow as tf

from external.nets import mobilenet_v1

slim = tf.contrib.slim


def mobilenet(inputs, num_classes, is_training,global_pool =True,data_format="channels_first"):
  with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
    logits, end_points = mobilenet_v1.mobilenet_v1(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions
