# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 20:49 2019
@author: kaiden
"""

"""Generate tfrecord file from images.
Example Usage:
---------------
python3 generate_tfrecord.py \
    --images_dir: Path to images (directory).
    --annotation_path: Path to annotatio's .txt file.
    --output_path: Path to .record.
    --resize_side_size: Resize images to fixed size.
"""

import io
import tensorflow as tf
import pandas as pd
from PIL import Image
import json
import os

flags = tf.app.flags

flags.DEFINE_string('images_dir', 
                    '', 
                    '')
flags.DEFINE_string('annotation_path', 
                    '',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path', 
                    '/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_integer('resize_side_size', None, 'Resize images to fixed size.')

FLAGS = flags.FLAGS


def provide(annotation_path=None, images_dir=None):
    """Return image_paths and class labels.
    
    Args:
        annotation_path: Path to an anotation's .csv file.
        images_dir: Path to images directory.
            
    Returns:
        image_files: A list containing the paths of images.
        annotation_dict: A dictionary containing the class labels of each 
            image.
            
    Raises:
        ValueError: If annotation_path does not exist.
    """
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')
        
    annotation_json = open(annotation_path, 'r')
    annotation_list = json.load(annotation_json)
    annotation_dict = {}
    
    anns_df = pd.DataFrame(annotation_list['annotations'])[['image_id','category_id']]
    img_df = pd.DataFrame(annotation_list['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
    df= pd.merge(img_df, anns_df, on='image_id')
    df['category_id']=df['category_id'].astype(int)

    df['file_name'] = images_dir + df['file_name']

    annotation_dict = dict(zip(df['file_name'], df['category_id']))

    return annotation_dict

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path, label, resize_size=None):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size
    
    # Resize
    if resize_size is not None:
        if width > height:
            width = int(width * resize_size / height)
            height = resize_size
        else:
            width = resize_size
            height = int(height * resize_size / width)
        image = image.resize((width, height), Image.ANTIALIAS)
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


def generate_tfrecord(annotation_dict, output_path, resize_size=None):
    num_valid_tf_example = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print('%s does not exist.' % image_path)
            continue
        tf_example = create_tf_example(image_path, label, resize_size)
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1
        
        if num_valid_tf_example % 100 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)
    
    
def main(_):
    images_dir = FLAGS.images_dir
    annotation_path = FLAGS.annotation_path
    record_path = FLAGS.output_path
    resize_size = FLAGS.resize_side_size
    
    annotation_dict = provide(annotation_path, images_dir)
    
    generate_tfrecord(annotation_dict, record_path, resize_size)
    
    
if __name__ == '__main__':
    tf.app.run()