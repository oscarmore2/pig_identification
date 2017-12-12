import numpy as np
import tensorflow as tf

def init_weights(name, shape):
	"""
	Handy helper function for initializing the weights of a layer.

	Performs He. et al. initilization as described in [1].

	References
	----------
	[1] - https://arxiv.org/abs/1502.01852
	"""
	init = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable(name, shape, tf.float32, init)
	return W

def init_bias(name, shape, trans=False):
	"""
	Handy helper function for initializing the biases of a layer.

	Performs zero bias initialization.
	"""
	init = tf.zeros_initializer
	b = tf.get_variable(name, shape, tf.float32, init)

	if trans:
		x = np.array([[1., 0, 0], [0, 1., 0]])
		x = x.astype('float32').flatten()
		b = tf.Variable(initial_value=x)

	return b

def Conv2D(input_tensor, input_shape, filter_size, num_filters, strides=1, name=None):
	"""
	Handy helper function for convnets.

	Performs 2D convolution with a default stride of 1. The kernel has shape
	filter_size x filter_size with num_filters output filters.
	"""
	shape = [filter_size, filter_size, input_shape, num_filters]

	# initialize weights and biases of the convolution
	W = init_weights(name=name+'_W' , shape=shape)
	b = init_bias(name=name+'_b', shape=shape[-1])

	conv = tf.nn.conv2d(input_tensor, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
	conv = tf.nn.bias_add(conv, b)
	return conv

def MaxPooling2D(input_tensor, k=2, use_relu=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs 2D max pool with a default stride of 2.
	"""
	pool = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

	if use_relu:
		pool = tf.nn.relu(pool)

	return pool

def BatchNormalization(input_tensor, phase, use_relu=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs batch normalization on the input tensor.
	"""
	normed = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=phase, scope=name)
	
	if use_relu:
		normed = tf.nn.relu(normed)
		
	return normed

def Residual_unit(input_, in_filter, out_filter, stride, option=0, name=None):
	x = BatchNormalization(input_, phase=True)
	x = tf.nn.relu(x)
	x = Conv2D(x, in_filter, 3, out_filter, name=name+'_resUnit_1')
	x = BatchNormalization(x, phase=True)
	x = tf.nn.relu(x)
	x = Conv2D(x, in_filter, 3, out_filter, name=name+'_resUnit_2')

	if in_filter != out_filter:
		if option == 0:
			difference = out_filter - in_filter
			left_pad = difference / 2
			right_pad = difference - left_pad
			identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
			return x + identity
		else:
			print ('Not implemented error')
			return None
	else:
		return x+input_

def batch_normalization_layer (input_layer, dimension):
	'''
	Helper function to do batch normalziation
	:param input_layer: 4D tensor
	:param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
	:return: the 4D tensor after being normalized
	'''
	mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
	beta = tf.get_variable('beta', dimension, tf.float32,
							   initializer=tf.constant_initializer(0.0, tf.float32))
	gamma = tf.get_variable('gamma', dimension, tf.float32,
								initializer=tf.constant_initializer(1.0, tf.float32))
	bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

	return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''

	out_channel = filter_shape[-1]
	filter = create_variables(name='conv', shape=filter_shape)

	conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
	bn_layer = batch_normalization_layer(conv_layer, out_channel)

	output = tf.nn.relu(bn_layer)
	return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
	'''
	A helper function to batch normalize, relu and conv the input layer sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
	'''

	in_channel = input_layer.get_shape().as_list()[-1]

	bn_layer = batch_normalization_layer(input_layer, in_channel)
	relu_layer = tf.nn.relu(bn_layer)

	filter = create_variables(name='conv', shape=filter_shape)
	conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
	return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
	'''
	Defines a residual block in ResNet
	:param input_layer: 4D tensor
	:param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
	:param first_block: if this is the first residual block of the whole network
	:return: 4D tensor.
	'''
	input_channel = input_layer.get_shape().as_list()[-1]

	# When it's time to "shrink" the image size, we use stride = 2
	if input_channel * 2 == output_channel:
		increase_dim = True
		stride = 2
	elif input_channel == output_channel:
		increase_dim = False
		stride = 1
	else:
		raise ValueError('Output and input channel does not match in residual blocks!!!')

	# The first conv layer of the first residual block does not need to be normalized and relu-ed.
	with tf.variable_scope('conv1_in_block'):
		if first_block:
			filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
			conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
		else:
			conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

	with tf.variable_scope('conv2_in_block'):
		conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

	# When the channels of input layer and conv2 does not match, we add zero pads to increase the
	#  depth of input layers
	if increase_dim is True:
		pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
									  strides=[1, 2, 2, 1], padding='VALID')
		padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
																	 input_channel // 2]])
	else:
		padded_input = input_layer

	output = conv2 + padded_input
	return output


def Flatten(layer):
	"""
	Handy function for flattening the result of a conv2D or
	maxpool2D to be used for a fully-connected (affine) layer.
	"""
	#print('layer')
	#print(tf.shape(layer))
	layer_shape = layer.get_shape().as_list()
	#print('Layer_shape')
	#print(layer_shape.as_list())
	# num_features = tf.reduce_prod(tf.shape(layer)[1:])
	num_features = np.prod(layer_shape[1:])
	#print('num_features')
	#print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

def Dense(input_tensor, num_inputs, num_outputs, use_relu=True, trans=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs an affine layer (fully-connected) on the input tensor.
	"""
	shape = [num_inputs, num_outputs]

	# initialize weights and biases of the affine layer
	W = init_weights(name=name+'_W' ,shape=shape)
	b = init_bias(name=name+'_b', shape=shape[-1], trans=trans)

	fc = tf.matmul(input_tensor, W, name=name) + b

	if use_relu:
		fc = tf.nn.relu(fc)

	return fc

def theta_bias(name):
	with tf.variable_scope(name):
		x = np.array([[1., 0, 0], [0, 1., 0]])
		x = x.astype('float32').flatten()
		return tf.Variable(initial_value=x)
