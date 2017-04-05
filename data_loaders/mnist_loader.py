import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
old_ld_path = os.environ["LD_LIBRARY_PATH"]
os.environ["LD_LIBRARY_PATH"]='/usr/lib/x86_64-linux-gnu/:%s' % old_ld_path

import sys
sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_local/lib/python2.7/site-packages/')


import cv2
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class mnist_loader:
	def __init__(self, data_directory, shape=(32, 32), intensity_bins=5):
		self.loader = input_data.read_data_sets(data_directory, one_hot=False)
		self.shape = shape
		self.intensity_bins = intensity_bins

	def train_next_batch(self, batch_size):
		batch_in, label = self.loader.train.next_batch(batch_size)
		return batch_in
#		batch = np.zeros((batch_size, np.prod(self.shape)*self.intensity_bins), dtype='f')
#		for i in range(batch_size):
#			img = cv2.resize(batch_in[i, ...].reshape((28, 28)), (self.shape[0], self.shape[1]))
#			batch[i, ...] = self.__get_1hotvec(img) 
#		return batch

	def test_next_batch(self, batch_size):
		batch_in, label = self.loader.test.next_batch(batch_size)
		return batch_in
#		batch = np.zeros((batch_size, np.prod(self.shape)*self.intensity_bins), dtype='f')
#		for i in range(batch_size):
#			img = cv2.resize(batch_in[i, ...].reshape((28, 28)), (self.shape[0], self.shape[1]))
#			batch[i, ...] = self.__get_1hotvec(img) 
#		return batch

	def __get_1hotvec(self, img_ch):
		bins = np.linspace(0., 1., self.intensity_bins)
		binvec = np.digitize(img_ch.reshape(-1), bins)-1
		vec_1hot = np.zeros((binvec.shape[0], self.intensity_bins), dtype='f')
		vec_1hot[range(0, binvec.shape[0]), binvec] = 1.
		return vec_1hot.reshape(-1)

	def save_output(self, net_op, epoch, itr_id, prefix, batch_size, out_directory, num_cols=16, net_recon_const=None, cluster=False):
#		net_op = net_op_ch.reshape((batch_size, self.shape[0]*self.shape[1]*self.intensity_bins))
		num_rows = np.int_(np.ceil((batch_size*1.)/num_cols))
		out_img = np.zeros((num_rows*self.shape[0], num_cols*self.shape[1]), dtype='uint8')
		c = 0
		r = 0
#		bins = np.linspace(0., 1., self.intensity_bins)
		for i in range(batch_size):
			if(i % num_cols == 0 and i > 0):
				r = r + 1
				c = 0
#			img1hot_mat = net_op[i, ...].reshape(self.shape[0], self.shape[1], self.intensity_bins)
#			imgbinid_mat =  np.argmax(img1hot_mat, axis=2)
#			out_img[r*self.shape[0]:(r+1)*self.shape[0], c*self.shape[1]:(c+1)*self.shape[1]] = np.uint8(255.*np.round(bins[imgbinid_mat]))
			out_img[r*self.shape[0]:(r+1)*self.shape[0], c*self.shape[1]:(c+1)*self.shape[1]] = np.uint8(np.round(255.*net_op[i, ...].reshape(self.shape[0], self.shape[1])))
			c = c+1
		out_fn = '%s/%s_%06d_%06d.png' % (out_directory, prefix, epoch, itr_id)
		print('[DEBUG] Writing output image: %s' % out_fn)
		cv2.imwrite(out_fn, out_img)

#if __name__ == '__main__':
#	dl = mnist_loader('/data/ardeshp2/MNIST/')
#	batch = dl.train_next_batch(32)
#	dl.save_output(batch.reshape(-1, 2), 0, 0, 'debug', 32, './')	
