import tensorflow as tf
import numpy as np
#from get2d_deconv_output_size import get2d_deconv_output_size
from batch_norm_aiuiuc import batch_norm_aiuiuc
from tensorflow.python.framework import tensor_shape
class layer_factory:
	def __init__(self):
		pass
	def weight_variable(self, name, shape=None, mean=0., stddev=.0001, gain=np.sqrt(2)):
		if(shape == None):
			return tf.get_variable(name)
#		#Adaptive initialize based on variable shape
#		if(len(shape) == 4):
#			stddev = (1.0 * gain) / np.sqrt(shape[0] * shape[1] * shape[3])
#		else:
#			stddev = (1.0 * gain) / np.sqrt(shape[0])
		return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))
	
	def bias_variable(self, name, shape=None, constval=.0001):
		if(shape == None):
			return tf.get_variable(name)
		return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(constval))
	def conv2d(self, x, W, stride=1, padding='SAME'):
		return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
	def batch_norm_aiuiuc_wrapper(self, x, train_phase, name, reuse_vars):
		bn = batch_norm_aiuiuc(x, train_phase, decay=.95, epsilon=1e-4, \
			name=name, reuse_vars=reuse_vars)
		output = bn.output
		return output
#	def deconv2d(self, x, W, stride=1, padding='SAME'):
#		input_height = x.get_shape()[1]
#		input_width = x.get_shape()[2]
#		filter_height = W.get_shape()[0]
#		filter_width = W.get_shape()[1]
#		out_rows, out_cols = get2d_deconv_output_size(input_height, input_width, 
#			filter_height, filter_width, stride, stride, padding)
#		out_batch = tensor_shape.as_dimension(x.get_shape()[0]).value
#		out_depth = tensor_shape.as_dimension(W.get_shape()[2]).value
#		out_shape = [out_batch, out_rows, out_cols, out_depth]
#		return tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, stride, stride, 1], padding=padding)
