from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import tensorflow as tf

slim = tf.contrib.slim


class BasicBlock(object):
    expansion = 1

    def __init__(self, name, filters, stride=1, training=True, downsample=None):
        self.name = '{}/BasicBlock'.format(name)
        self.filters = filters
        self.downsample = downsample
        self.stride = stride
        self.training = training
        self.reuse = tf.AUTO_REUSE

    def forward(self, x):
        inputs = tf.convert_to_tensor(x)
        with tf.variable_scope(self.name, reuse=self.reuse):
            out = None
            with slim.arg_scope([slim.conv2d],num_outputs=self.filters,
                                kernel_size=[3, 3],
                                stride=self.stride,
                                padding='same'):
                with tf.variable_scope('conv1'):
                    out = slim.conv2d(inputs=inputs)
                    out = slim.batch_norm(out, activation_fn=tf.nn.relu, scope='bn1')

                with tf.variable_scope('conv2'):
                    out = slim.conv2d(inputs=out)
                    out = slim.batch_norm(out, activation_fn=tf.nn.relu, scope='bn2')

            if self.downsample is not None:
                inputs = self.downsample._fn(inputs)

            out += inputs
            out = tf.nn.relu(out)

            return out


class Bottleneck(object):
    expansion = 4

    def __init__(self, name, filters, stride=1, training=True, downsample=None):
        self.name = '{}/Bottleneck'.format(name)
        self.filters = filters
        self.downsample = downsample
        self.stride = stride
        self.reuse = tf.AUTO_REUSE
        self.training = training

    def forward(self, x):
        inputs = tf.convert_to_tensor(x)
        with tf.variable_scope(self.name, reuse=self.reuse):
            out = None
            with tf.variable_scope('conv1'):
                out = slim.conv2d(inputs=x,
                                            num_outputs = self.filters,
                                            kernel_size=[1, 1],
                                            padding='valid')

                out = slim.batch_norm(out, activation_fn=tf.nn.relu, scope='bn1')

            with tf.variable_scope('conv2'):
                out = slim.conv2d(inputs=out,
                                            num_outputs=self.filters,
                                            kernel_size=[3, 3],
                                            stride=self.stride,
                                            padding='same')
                out = slim.batch_norm(out, activation_fn=tf.nn.relu, scope='bn1')
            with tf.variable_scope('conv3'):
                out = slim.conv2d(inputs=out,
                                            num_outputs=self.filters * self.expansion,
                                            kernel_size=[1, 1],
                                            padding='valid')
                out = slim.batch_norm(out, scope='bn1')

            if self.downsample is not None:
                inputs = self.downsample._fn(inputs)

            out += inputs
            out = tf.nn.relu(out)

            return out


class DownSample(object):
    def __init__(self, filters, kernel_size, stride, padding='valid', training=True, scope=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.training = training
        self.scope = scope
        self.reuse = tf.AUTO_REUSE

    def _fn(self, x):
        with tf.variable_scope(self.scope,reuse=self.reuse):
            out = slim.conv2d(inputs=x,
                                    num_outputs=self.filters,
                                    kernel_size=[self.kernel_size, self.kernel_size],
                                    stride=self.stride,
                                    padding=self.padding)

            out = slim.batch_norm(out)
            return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionModule(object):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, training=True, scope=None):
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.training = training
        self.scope = scope
        self.reuse = tf.AUTO_REUSE

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _branch(self, input, branch_index, block, num_blocks, num_channels,
                stride=1, scope=None):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = DownSample(filters=num_channels[branch_index] * block.expansion,
                                    kernel_size=1, stride=stride, padding='valid',
                                    training=self.training, scope='{}/{}/downsample'.format(self.scope, scope))
        block_layer1 = block(name='{}/{}/block1'.format(self.scope, scope),
                             filters=num_channels[branch_index],
                             stride=stride,
                             training=self.training,
                             downsample=downsample)
        block_layer2 = block(name='{}/{}/block2'.format(self.scope, scope),
                             filters=num_channels[branch_index],
                             stride=stride,
                             training=self.training)

        out = block_layer1.forward(input)
        for i in range(1, num_blocks[branch_index]):
            out = block_layer2.forward(out)

        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion

        return out

    def conv_layer(self, input, filters, kernel_size, stride, padding, scope=None, activation=True):
        with tf.variable_scope(scope,reuse = self.reuse):
            out = slim.conv2d(inputs=input,
                                    num_outputs = filters,
                                    kernel_size=[kernel_size, kernel_size],
                                    stride=stride,
                                    padding=padding)
            out = slim.batch_norm(out, scope='bn')
            if activation:
                out = tf.nn.relu(out)
            return out

    def upsample_layer(self, input, shape, scope=None):
        with tf.variable_scope(scope):
            out = tf.image.resize_nearest_neighbor(input, size=shape)
            return out

    def relu_layer(self,input, scope):
        return tf.nn.relu(input)

    def _fuse_layers(self, x, multi_scale_output=False):
        if self.num_branches == 1:
            return x

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        out_fuse = []
        for i in range(num_branches if multi_scale_output else 1):
            out = x[0]
            if i > 0:
                for k in range(i):
                    if k == i - 1:
                        num_outchannels = num_inchannels[i]
                        out = self.conv_layer(out,
                                              filters=num_outchannels,
                                              kernel_size=3,
                                              stride=2,
                                              padding='same',
                                              scope='{}/fuse_conv{}_0_{}'.format(self.scope, i, k),
                                              activation=False)
                    else:
                        num_outchannels = num_inchannels[0]
                        out = self.conv_layer(out,
                                              filters=num_outchannels,
                                              kernel_size=3,
                                              stride=2,
                                              padding='same',
                                              scope='{}/fuse_conv{}_0_{}'.format(self.scope, i, k),
                                              activation=True)

            for j in range(1, num_branches):
                y = x[j]
                if j > i:
                    y = self.conv_layer(y,
                                        filters=num_inchannels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding='valid',
                                        scope='{}/fuse_conv{:d}_{:d}'.format(self.scope, i, j),
                                        activation=False)

                    y = self.upsample_layer(y, [(2 ** (j - i)) * tf.shape(y)[1], (2 ** (j - i)) * tf.shape(y)[2]],
                                            scope='{}/fuse_upsample{:d}_{:d}'.format(self.scope, i, j))
                    out = y + out
                elif j == i:
                    out = out + x[j]
                else:
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.num_inchannels[i]

                            y = self.conv_layer(y,
                                                filters=num_outchannels_conv3x3,
                                                kernel_size=3,
                                                stride=2,
                                                padding='same',
                                                scope='{}/fuse_conv{:d}_{:d}_{:d}'.format(self.scope, i, j, k),
                                                activation=False)
                        else:
                            num_outchannels_conv3x3 = self.num_inchannels[j]
                            y = self.conv_layer(y,
                                                filters=num_outchannels_conv3x3,
                                                kernel_size=3,
                                                stride=2,
                                                padding='same',
                                                scope='{}/fuse_conv{:d}_{:d}_{:d}'.format(self.scope, i, j, k),
                                                activation=True)
                    out = out + y
            out_fuse.append(self.relu_layer(out,'{}/fuse_relu_{}'.format(self.scope, i)))

        return out_fuse

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x, multi_scale_output):
        if self.num_branches == 1:
            return [self._branch(x[0], 0, self.blocks, self.num_blocks, self.num_channels)]

        for i in range(self.num_branches):
            x[i] = self._branch(x[i], i, self.blocks, self.num_blocks, self.num_channels, scope='/branch{}'.format(i))

        x_fuse = self._fuse_layers(x, multi_scale_output)

        return x_fuse


class HighResolutionNet(object):

    def __init__(self, cfg, istraining, **kwargs):

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        self.num_channels1 = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        self.num_channels1 = [
            self.num_channels1[i] * block.expansion for i in range(len(self.num_channels1))]

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        self.num_channels2 = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        self.num_channels2 = [
            self.num_channels2[i] * block.expansion for i in range(len(self.num_channels2))]

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        self.num_channels3 = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        self.num_channels3 = [
            self.num_channels3[i] * block.expansion for i in range(len(self.num_channels3))]

        self.training = istraining
        self.end_points = {}
        self.reuse = tf.AUTO_REUSE

    def conv_layer(self, input, filters, kernel_size, stride, padding, scope=None):

        with tf.variable_scope(scope, reuse=self.reuse):
            out = slim.conv2d(inputs=input,
                                    num_outputs=filters,
                                    kernel_size=[kernel_size, kernel_size],
                                    stride=stride,
                                    padding=padding)
            out = slim.batch_norm(out, activation_fn=tf.nn.relu,is_training=self.training, scope='bn1')
            return out

    def prediction_fn(self, x, scope=None):
        with tf.variable_scope(scope):
            return tf.nn.softmax(x)


    def _image2head(self, input, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = DownSample(filters=out_channels,
                                         kernel_size=3, stride=2, padding='same',
                                         training=self.training, scope='image2head{:d}'.format(i))

            downsamp_modules.append(downsamp_module)

        out = self._make_layer(input[0], head_block,
                               head_channels[0],
                               1,
                               stride=1,
                               scope='image2head_incre_layer0')

        for i in range(len(pre_stage_channels) - 1):
            y = self._make_layer(input[i + 1],
                                 head_block,
                                 head_channels[i + 1],
                                 1,
                                 stride=1,
                                 scope='image2head_incre_layer{:d}'.format(i+1))
            out = downsamp_modules[i]._fn(out) + y

        
        out = self.conv_layer(out,
                              filters=2048,
                              kernel_size=1,
                              stride=1,
                              padding='valid',
                              scope='head_final')

        return out

    def _transition_layer(
            self, input, num_channels_pre_layer, num_channels_cur_layer, scope=None):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        out = []
        for i in range(num_branches_cur):
            out_branch = input[-1]
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    out.append(self.conv_layer(out_branch, num_channels_cur_layer[i],
                                               3, 1, 'same', '{}/conv{}'.format(scope, i)))
                else:
                    out.append(input[i])
            else:
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    out_branch = self.conv_layer(out_branch, outchannels,
                                                 3, 2, 'same', '{}/conv{}_{}'.format(scope, i, j))
                out.append(out_branch)

        return out

    def _make_layer(self, input, block, filters, blocks, stride=1, scope=None):
        downsample = None
        input_shape = input.shape[-1]
        if stride != 1 or input_shape != filters * block.expansion:
            downsample = DownSample(filters=filters * block.expansion,
                                    kernel_size=1, stride=stride, padding='valid',
                                    training=self.training, scope='{}/downsamp'.format(scope))
        out = None
        block_layer1 = block(name='{}/block1'.format(scope), filters=filters,
                             stride=stride,
                             training=self.training,
                             downsample=downsample)
        block_layer2 = block(name='{}/block2'.format(scope), filters=filters,
                             stride=stride,
                             training=self.training)

        out = block_layer1.forward(input)
        for i in range(1, blocks):
            out = block_layer2.forward(out)

        return out

    def _stage(self, input, layer_config, num_inchannels,
               multi_scale_output=True, scope=None):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        out = input
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules = HighResolutionModule(num_branches,
                                           block,
                                           num_blocks,
                                           num_inchannels,
                                           num_channels,
                                           fuse_method,
                                           self.training,
                                           scope='Modules{}'.format(i))

            out = modules.forward(out, reset_multi_scale_output)
            num_inchannels = modules.get_num_inchannels()

        return out, num_inchannels

    def forward(self, x,num_classes,dropout_keep_prob=0.8):

        inputs = tf.convert_to_tensor(x)
        out = self.conv_layer(inputs, 64, 3, 2, 'same', 'conv0')
        out = self.conv_layer(out, 64, 3, 2, 'same', 'conv1')

        out = self._make_layer(out, Bottleneck, 64, 4, scope='make_layer0')
        out_list = []
        out_list.append(out)
        out_list = self._transition_layer(out_list, [256], self.num_channels1, scope='transition_layer0')
        y_list, pre_stage_channels = self._stage(out_list, self.stage2_cfg, self.num_channels1)

        out_list = self._transition_layer(y_list, pre_stage_channels, self.num_channels2, scope='transition_layer1')
        y_list, pre_stage_channels = self._stage(out_list, self.stage3_cfg, self.num_channels2)

        out_list = self._transition_layer(y_list, pre_stage_channels, self.num_channels3, scope='transition_layer2')
        y_list, pre_stage_channels = self._stage(out_list, self.stage4_cfg, self.num_channels3)

        # Classification Head
        y = self._image2head(y_list, pre_stage_channels)

        with tf.variable_scope('Logits',reuse=self.reuse):
            kernel_size = y.get_shape()[1:3]
            if kernel_size.is_fully_defined():
                y = slim.avg_pool2d(y, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
            else:
                y = tf.reduce_mean(y, [1, 2], keep_dims=True, name='global_pool')
            self.end_points['global_pool'] = y
            y = slim.flatten(y)
            y = slim.dropout(y, dropout_keep_prob, is_training= self.training,
                           scope='Dropout')
            self.end_points['PreLogits'] = y
            y = slim.fully_connected(y, num_classes, activation_fn=None,
                                      scope='Logits')
        self.end_points['Logits'] = y
        self.end_points['Predictions'] = self.prediction_fn(y, 'Predictions')

        return y, self.end_points

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')


def hr_resnet18(pretrained=False, is_training=True, **kwargs):
    cfg = {'MODEL': {
        'EXTRA': {
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [18, 36],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [18, 36, 72],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [18, 36, 72, 144],
                'FUSE_METHOD': 'SUM',
            }
        }
    }
    }
    model = HighResolutionNet(cfg, is_training, **kwargs)
    model.init_weights()
    return model


def hr_resnet48(pretrained=False, is_training=True, **kwargs):
    cfg = {'MODEL': {
        'EXTRA': {
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [48, 96],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192, 384],
                'FUSE_METHOD': 'SUM',
            }
        }
    }
    }
    model = HighResolutionNet(cfg, is_training, **kwargs)
    # model.init_weights()
    return model


def hr_resnet64(pretrained=False, is_training=True, **kwargs):
    cfg = {'MODEL': {
        'EXTRA': {
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [64, 128],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [64, 128, 256],
                'FUSE_METHOD': 'SUM',
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [64, 128, 256, 512],
                'FUSE_METHOD': 'SUM',
            }
        }
    }
    }
    model = HighResolutionNet(cfg, is_training, **kwargs)
    # model.init_weights()
    return model

def hrnet_v2_arg_scope(
    weight_decay=0.00004,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
    batch_norm_scale=False):
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,  # Use fused batch norm if possible.
        'scale': batch_norm_scale,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d],normalizer_params=batch_norm_params) as scope:
      return scope

def main(_):
    tf.reset_default_graph()
    model = hr_resnet48()
    x = tf.placeholder(tf.float32, [24, 224, 224, 3])
    y = model.forward(x, 1000)


if __name__ == '__main__':
    tf.app.run()