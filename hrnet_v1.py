from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import tensorflow as tf
from tensorflow.python.ops import math_ops

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(object):
	expansion = 1
    def __init__(self, name,filters, stride=1, training=True, downsample=None):
        super(BasicBlock, self).__init__()
        self.name = name.append('_BasicBlock')
        self.filters = filters
        self.downsample = downsample
        self.stride = stride
        self.training = training
        self.reuse = False

    def forward(self, x):
        inputs = tf.convert_to_tensor(x)
        with tf.variable_scope(self.name, reuse=self.reuse):
            out = None
            with tf.variable_scope('conv1'):
                out = tf.layers.conv2d(inputs=inputs,
                            filters=self.filters,
                            kernel_size=[3,3],
                            stride=[self.stride,self.stride],
                            padding='same' )
                out = tf.layers.relu(tf.layers.batch_normalization(out, training=self.training),name='outputs')
            
            with tf.variable_scope('conv2'):
                out = tf.layers.conv2d(inputs=out,
                            filters=self.filters,
                            kernel_size=[3,3],
                            stride=[self.stride,self.stride],
                            padding='same' )
                out = tf.layers.relu(tf.layers.batch_normalization(out, training=self.training),name='outputs')

            if self.downsample is not None:
                inputs = self.downsample._fn(inputs)

            out += inputs
            out = tf.layers.relu(out,training = self.training)

            return out


class Bottleneck(object):
	expansion = 4
    def __init__(self,name, filters, stride=1,training=True, downsample=None):
        super(Bottleneck, self).__init__()
        self.name = name.append('_Bottleneck')
        self.filters = filters
        self.downsample = downsample
        self.stride = stride
        self.reuse = False
        self.training = training

    def forward(self, x):
        inputs = tf.convert_to_tensor(x)
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                out = tf.layers.conv2d(inputs=x,
                            filters=self.filters,
                            kernel_size=[1,1],
                            padding='valid' )

                out = tf.layers.relu(tf.layers.batch_normalization(out, training=self.training),name='outputs')

            with tf.variable_scope('conv2'):
                out = tf.layers.conv2d(inputs=out,
                            filters=self.filters,
                            kernel_size=[3,3],
                            stride=[self.stride,self.stride],
                            padding='same' )
                out = tf.layers.relu(tf.layers.batch_normalization(out, training=self.training),name='outputs')
            with tf.variable_scope('conv3'):
                out = tf.layers.conv2d(inputs=out,
                            filters=self.filters * self.expansion,
                            kernel_size=[1,1],
                            padding='valid' )
                out = tf.layers.batch_normalization(out, training=self.training)
        

            if self.downsample is not None:
                inputs = self.downsample._fn(inputs)

            out += inputs
            out = tf.layers.relu(out,training = self.training)

            return out

class DownSample(object):
    def _init_(self, filters, kernel_size, stride, padding='valid', training=True, scope=None):
        super(DownSample, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.training = training
        self.scope = scope.append('_DownSample')

    def _fn(self, x):
        with tf.variable_scope(self.scope):
            out = tf.layers.conv2d(inputs=x,
                            filters=self.filters,
                            kernel_size=[self.kernel_size,self.kernel_size],
                            padding=self.padding)

            out = tf.layers.batch_normalization(out, training=self.training)
            return out

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
class HighResolutionModule(object):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, scope=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.scope = scope

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
                                   training=self.training, scope=self.scope.append(scope).append('_downsample'))
        block_layer1 =  blocks_dict[block](name=self.scope.append(scope).append('_block1'),
                                    filters=num_channels[branch_index],
                                    stride=stride,
                                    training=self.training, 
                                    downsample=downsample)
        block_layer2 = blocks_dict[block](name=self.scope.append(scope).append('_block2'),
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
        with tf.variable_scope(scope):
            out = tf.layers.Conv2d(inputs=input,
                            filters=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
            out = tf.layers.batch_normalization(out, training=self.training)
            if activation:
                out = tf.layers.relu(out,name='outputs')
            return out
    def upsample_layer(self,input,shape,scope=None):
        with tf.variable_scope(scope):
            out = tf.image.resize_nearest_neighbor(input,size=shape)
            return out
    def relu_layer(input,scope):
        with tf.variable_scope(scope):
            return tf.layers.relu(input)

    def _fuse_layers(self,x,multi_scale_output=False):
        if self.num_branches == 1:
            return x

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        out_fuse = []
        for i in range(num_branches if multi_scale_output else 1):
            out = x[0]
            if i > 0:
                for k in range(i):
                    if k== i-1:
                        num_outchannels = num_inchannels[i]
                        out = self.conv_layer(out,
                                            filters=num_outchannels,
                                            kernel_size=[3, 3],
                                            stride=[2, 2],
                                            padding='same',
                                            scope=self.scope.append('_fuse_conv{}_0_{}'.format(i,k)),
                                            activation=False)
                    else:
                        num_outchannels = num_inchannels[0]
                        out = self.conv_layer(out,
                                            filters=num_outchannels,
                                            kernel_size=[3, 3],
                                            stride=[2, 2],
                                            padding='same',
                                            scope=self.scope.append('_fuse_conv{}_0_{}'.format(i,k)),
                                            activation=True)

            for j in range(1, num_branches):
                y = x[j]
                if j > i:
                    y = self.conv_layer(y,
                                        filters=num_inchannels[i],
                                        kernel_size=[1, 1],
                                        stride=[1, 1],
                                        padding='valid',
                                        scope=self.scope.append('fuse_conv{:d}_{:d}'.format(i, j)),
                                        activation=False)
                    
                    y = self.upsample_layer(y, [(2**(j-i)) * tf.shape(out)[1], (2**(j-i))*tf.shape(out)[2]],
                                    scope=self.scope.append('_fuse_upsample{:d}_{:d}'.format(i, j)))
                    out = y + out
                elif j == i:
                     out = out + x[j]
                else: 
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.num_inchannels[i]

                            y = self.conv_layer(y,
                                            filters = num_outchannels_conv3x3,
                                            kernel_size = [3,3],
                                            stride = [2,2],
                                            padding = 'same',
                                            scope = self.scope.append('_fuse_conv{:d}_{:d}_{:d}'.format(i, j, k)),
                                            activation = False)
                        else:
                            num_outchannels_conv3x3 = self.num_inchannels[j]
                            y = self.conv_layer(y,
                                            filters = num_outchannels_conv3x3,
                                            kernel_size = [3,3],
                                            stride = [2,2],
                                            padding = 'same',
                                            scope = self.scope.append('_fuse_conv{:d}_{:d}_{:d}'.format(i, j, k)),
                                            activation = True)
                    out = out + y
            out_fuse.append(self.relu_layer(out, self.scope.append('_fuse_relu_{%d}'.format(i))))

        return out_fuse

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x, multi_scale_output):
        if self.num_branches == 1:
            return [self._branche(x[0], 0, self.blocks, self.num_blocks, self.num_channels)]
        
        for i in range(self.num_branches):
            x[i] = self._branche(x[i], i, self.blocks, self.num_blocks, self.num_channels, scope='_branch{}'.format(i))

        x_fuse = self._fuse_layers(x, multi_scale_output)
        

        return x_fuse



class HighResolutionNet(object):

    def __init__(self, cfg, istraining, **kwargs):
        super(HighResolutionNet, self).__init__()


        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        self.num_channels1 = self.stage2_cfg['NUM_CHANNELS']
        self.block1 = blocks_dict[self.stage2_cfg['BLOCK']]
        self.num_channels1 = [
            self.num_channels1[i] * self.block1.expansion for i in range(len(self.num_channels1))]


        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        self.num_channels2 = self.stage3_cfg['NUM_CHANNELS']
        self.block2 = blocks_dict[self.stage3_cfg['BLOCK']]
        self.num_channels2 = [
            self.num_channels2[i] * self.block2.expansion for i in range(len(self.num_channels2))]
        
        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        self.num_channels3 = self.stage4_cfg['NUM_CHANNELS']
        self.block4 = blocks_dict[self.stage4_cfg['BLOCK']]
        self.num_channels3 = [
            self.num_channels3[i] * self.block4.expansion for i in range(len(self.num_channels3))]
        
        self.training = istraining
        self.end_points={}

    def conv_layer(self,input,filters,kernel_size,stride,padding,scope=None):
        with tf.variable_scope(scope):
            out = tf.layers.Conv2d(inputs=input,
                            filters=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
            out = tf.layers.relu(tf.layers.batch_normalization(out, training=self.training),name='outputs')
            return out

    def prediction_fn(self, x, scope=None):
        with tf.variable_scope(scope):
            return tf.nn.softmax(x)

    def dropout(self, x, scope=None):
        with tf.variable_scope(scope):
            return tf.layers.dropout(x)

    def _image2head(self,input, pre_stage_channels, num_classes):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        

         # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = DownSample(filters=out_channels,
                                    kernel_size=3, stride=2, padding='same',
                                   training=self.training, scope='image2head{:d}'.format(i))

            downsamp_modules.append(downsamp_module)


        out =  self._make_layer(input[0], head_block,
                                          head_channels[0],
                                          1,
                                          stride=1,
                                          scope= 'image2head_incre_layer0')

        for i in range(len(pre_stage_channels)-1):

            y = self._make_layer(input[i+1],
                                    head_block,
                                    head_channels[i+1],
                                    1,
                                    stride=1,
                                    scope='image2head_incre_layer{:d}'.format(i))
            out =  downsamp_modules[i]._fn(out) + y

        out = self.dropout(out, scope='Dropout')

        self.end_points['PreLogits'] = net

        out = self.conv_layer(out,
                            filters=num_classes,
                            kernel_size=1,
                            stride=1,
                            padding='valid',
                            scope = 'head_final')

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
                    out.append(self.conv_layer(out_branch, num_channels_cur_layer[i]
                        [3,3], [1, 1], 1, 'valid', scope.append('_conv{:d}'.format(i))))
                else:
                    out.append(input[i])
            else:
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    out_branch = self.conv_layer(out_branch, outchannels,
                        [3, 3], [2, 2], 1, 'valid', scope.append('_conv{:d}_{:d}'.format(i, j)))
                out.append(out_branch)

        return out

    def _make_layer(self, input,block, filters, blocks, stride=1, scope=None):
        downsample = None
        input_shape = input.shape[-1]
        if stride != 1 or input_shape != filters * block.expansion:
            downsample = DownSample(filters=filters * block.expansion,
                                    kernel_size=1, stride=stride, padding='valid',
                                   training=self.training, scope=scope.append('_downsamp'))
        out = None
        block_layer1 =  blocks_dict[block](name=scope.append('_block1'), filters=filters,
                                    stride=stride,
                                    training=self.training, 
                                    downsample=downsample)
        block_layer2 = blocks_dict[block](name=scope.append('_block2'), filters=filters,
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
                                           scope='Modules{}'.format(i))

            out = modules.forward(out, reset_multi_scale_output)
            num_inchannels = modules.get_num_inchannels()

        return out, num_inchannels


    def forward(self, x, num_classes):

        inputs = tf.convert_to_tensor(x)
        out = self.conv_layer(inputs, 64, [3, 3], [2, 2], 'same', 'conv0')
        out = self.conv_layer(out, 64, [3, 3], [2, 2], 'same', 'conv1')

        out = self._make_layer(out, 'Bottleneck', 64, 4, scope='make_layer0')

        out_list = self._transition_layer(out, [256], self.num_channels1, scope='transition_layer0')
        y_list, pre_stage_channels = self._stage(out_list, self.stage2_cfg, self.num_channels1)


        out_list = self._transition_layer(y_list, pre_stage_channels, self.num_channels2, scope='transition_layer1')
        y_list,pre_stage_channels= self._stage(out_list, self.stage3_cfg, self.num_channels2)

        
        out_list = self._transition_layer(y_list,pre_stage_channels,self.num_channels3, scope='transition_layer2')
        y_list,pre_stage_channels= self._stage(out_list, self.stage4_cfg,self.num_channels3)

        # Classification Head
        y = self._image2head(y_list, pre_stage_channels, num_classes)

        y = math_ops.reduce_mean(y, [1, 2], name='pool5', keepdims=True)
        self.end_points['Logits'] = y
        self.end_points['Predictions'] = self.prediction_fn(y,'Predictions')

        return y, self.end_points

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')

def hr_resnet18(pretrained=False, is_training = True, **kwargs):
    cfg = {'MODEL':{
                'EXTRA':{
                    'STAGE2':{
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS':[18,36],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE3':{
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4],
                        'NUM_CHANNELS':[18,36,72],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE4':{
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4,4],
                        'NUM_CHANNELS':[18,36,72,144],
                        'FUSE_METHOD': 'SUM',
                            }
                        }
                    }
            }
    model = HighResolutionNet(cfg, is_training, **kwargs)
    model.init_weights()
    return model

def hr_resnet48(pretrained=False, is_training = True, **kwargs):
    cfg = {'MODEL':{
                'EXTRA':{
                    'STAGE2':{
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS':[48, 96],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE3':{
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4],
                        'NUM_CHANNELS':[48, 96, 192],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE4':{
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4,4],
                        'NUM_CHANNELS':[48, 96, 192, 384],
                        'FUSE_METHOD': 'SUM',
                            }
                        }
                    }
            }
    model = HighResolutionNet(cfg, is_training,  **kwargs)
    #model.init_weights()
    return model

def hr_resnet64(pretrained=False, is_training=True, **kwargs):
    cfg = {'MODEL':{
                'EXTRA':{
                    'STAGE2':{
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS':[64, 128],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE3':{
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4],
                        'NUM_CHANNELS':[64, 128, 256],
                        'FUSE_METHOD': 'SUM',
                            },
                    'STAGE4':{
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4,4,4,4],
                        'NUM_CHANNELS':[64, 128, 256, 512],
                        'FUSE_METHOD': 'SUM',
                            }
                        }
                    }
            }
    model = HighResolutionNet(cfg, is_training, **kwargs)
    #model.init_weights()
    return model

